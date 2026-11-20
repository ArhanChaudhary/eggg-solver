//! Phase-1 move masks: per staging state, the moves that decrease its staging distance (and,
//! in the reduced table, those that keep it).
//!
//! Deep in the phase-1 tree almost every node is at exact staging distance = moves left, and
//! then the children to search are exactly the distance-decreasing moves: one read of the
//! node's mask replaces the ~20 table reads of its children (one cache line each; ~10x fewer
//! reads in all on the p1check workload, same endpoints). A node's mask is needed before its
//! children are known, so the search expands such nodes in batches, breadth-first, each mask
//! prefetched a round ahead (see `Search1::run_batch`).
//!
//! Two tables: the large one, indexed like the large staging table (735471 x 6435), 27 bits
//! per state (4 bytes, 18.9 GB); and the reduced one, indexed like the symmetry-reduced
//! staging table (15582 x 6435 states, each the image of the searched state under the
//! symmetry that takes its U/D mask to the class representative), with the masks over all 36
//! outer block moves: the searched state's 27-move mask is the stored mask with each move
//! conjugated by that symmetry (five byte lookups). Its entries (16 bytes, one cache line;
//! 1.6 GB) hold the masks of the moves that decrease and that keep the distance, and two
//! parity-viability bits (see [`MaskTable::add_viable`]): a node at distance = moves left
//! whose wing parity cannot end at an endpoint is dropped as soon as its entry is read, and
//! a node with slack (distance < moves left) has its children's exact distances from its own
//! entry, one read instead of one per child. The search then needs no staging table but for
//! the few nodes above the job prefixes (the symmetry-reduced one, 25 MB, does).
//!
//! For the per-pair searches (see [`crate::p1pat`]), the pair table holds the same for the
//! pattern distance: by the reduced pattern index (3.5e9 states), the 36-move masks of the
//! moves that decrease it (5 bytes, 17.5 GB).

use crate::comb::comb_unrank;
use crate::cube::{B, D, L, Perm, Turn};
use crate::moves::{MOVE_PERMS, N_MOVES};
use crate::phase1::{
    N_CLASSES_A, N_SPLIT, P1State, P1Tables, Staging, centre_syms, flag_masks, pdep,
};
use crate::simd::V32;
use crate::table::HugeBuf;

const MASK24: u32 = 0xff_ffff;
/// The default of [`MaskTable::slack_batch`].
pub const SLACK_BATCH: u8 = 8;
/// The 36 outer block moves: the solver's 27, then Dw, Lw, Bw (each power).
const N36: usize = 36;
/// The 36-move masks' bits of a reduced entry.
const MASK36: u64 = (1 << N36) - 1;
/// The first of a reduced entry's two viability bits (see [`MaskTable::add_viable`]).
const VIABLE_SHIFT: u32 = 36;
/// A reduced entry's successors (see [`MaskTable::add_succ`]), [`SUCC_BITS`] bits each: the
/// first two from bit [`SUCC_SHIFT0`] of the entry's first word (above the viability bits),
/// the next two from bit 36 of its second word (above the equal moves' mask), and in the
/// wide table's entries four more in each of their two extra words. Bit [`SUCC_FLAG`] of the
/// second word is set where they are stored, and bit [`SUCC_EQ`] where they are those of the
/// decreasing and the equal moves (else of the decreasing moves only).
const SUCC_BITS: u32 = 13;
const SUCC_SHIFT0: u32 = 38;
const SUCC_FLAG: u32 = 62;
const SUCC_EQ: u32 = 63;
/// The most successors of an entry: of the reduced table (16 bytes), of the wide one (32).
const SUCC_MAX: u32 = 4;
const SUCC_MAX_WIDE: u32 = 12;
/// Marks a batch node whose reduced index is not resolved yet (see `MaskTable::resolve`).
pub const UNRESOLVED: u8 = 0xff;
/// The wide quarter turns among the 36 moves: Uw, Rw, Fw, Dw, Lw, Bw and their inverses (the
/// moves that change the wing parity).
const WIDE_QUARTER36: u64 = crate::moves::WIDE_QUARTER as u64
    | (1 << 27)
    | (1 << 29)
    | (1 << 30)
    | (1 << 32)
    | (1 << 33)
    | (1 << 35);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaskKind {
    /// 27-bit masks by the large staging index (18.9 GB).
    Big,
    /// 36-bit masks and the viability bits by the reduced staging index (0.8 GB).
    Reduced,
    /// No table: masks computed from the staging table when needed (tests).
    Virtual,
    /// 36-bit masks of pattern-distance-decreasing moves by the reduced pattern index
    /// (5 bytes per entry, 17.5 GB), for the per-pair searches.
    Pair,
}

pub struct MaskTable {
    pub kind: MaskKind,
    /// Batches also read the masks of nodes one move from the end, so that only their
    /// staging last moves get the wing tests (`EGGG_P1_BATCH_LEAVES` set).
    pub batch_leaves: bool,
    /// Pending batch nodes that start a batch run (`EGGG_P1_BATCH`, default 32).
    pub batch: usize,
    /// With the reduced table, nodes with slack (distance < moves left) and at most this
    /// many moves left are expanded in batches too (`EGGG_P1_SLACK_BATCH`, default
    /// [`SLACK_BATCH`]; 0: none).
    pub slack_batch: u8,
    big: HugeBuf<u32>,
    /// The reduced table's entries: the decreasing moves' 36-bit mask with the viability bits
    /// above it, and the equal moves' mask (16 bytes, aligned: one cache line); in the wide
    /// table, each followed by 16 bytes of further successors.
    red: HugeBuf<[u64; 2]>,
    /// log2 of the 16-byte units per entry of `red` (1: the wide table).
    shift: u32,
    /// The pair table: 5 bytes per entry (and 3 bytes of padding).
    pair: HugeBuf<u8>,
    /// The pair table's entries are computed when read, from the pattern table (tests).
    lazy: bool,
    /// Per symmetry, per byte of a stored 36-bit mask: the searched state's 27-bit mask.
    conj: Vec<[[u32; 256]; 5]>,
    /// The batches take the children of nodes at distance = moves left from their entries'
    /// successors where stored (`EGGG_P1_SUCC=0`: never; see [`MaskTable::add_succ`]).
    pub use_succ: bool,
    /// Per symmetry s, per (searched) move m: the move of the 36 that m is on the symmetric
    /// image (the move conjugated by s; the bit of m in the conjugation lookups `conj`).
    conj_move: Vec<[u8; N_MOVES]>,
    /// Per class, per move k of the 36: the class info (class << 8 | symmetry) of the U/D
    /// mask of the class representative after k (see [`MaskTable::succ_child`]).
    succ_class: Vec<u32>,
    /// The product of symmetries a and b (first b, then a), at 48 a + b.
    sym_mul: Vec<u8>,
}

/// A batch node's successors (see [`MaskTable::succ`]).
#[derive(Clone, Copy)]
pub struct Succ {
    /// The moves (of the 36) whose successors are stored: the decreasing ones, and the equal
    /// ones if `eq`.
    list: u64,
    /// The successors in the order of their moves, four per word ([`SUCC_BITS`] each).
    ch: [u64; 3],
    /// The node's class times 36: its row of `succ_class`.
    row: usize,
    /// The equal moves' successors are stored too.
    pub eq: bool,
}

/// Centre gathers (positions 24.. fixed) of the 36 outer block moves.
fn moves36() -> [V32; N36] {
    std::array::from_fn(|m| {
        let src = if m < N_MOVES {
            MOVE_PERMS[m].centre_src
        } else {
            let face = [D, L, B][(m - N_MOVES) / 3];
            Perm::of_turn(Turn::face(face, 2, (m % 3) as u8 + 1)).centre_src
        };
        let mut v = V32(std::array::from_fn(|i| i as u8));
        v.0[..24].copy_from_slice(&src);
        v
    })
}

/// Flags (see `P1State::centre_flags`) of the state with U/D mask `ud` and R/L mask `rl`.
fn flags_of(ud: u32, rl: u32) -> V32 {
    let mut flags = V32::ZERO;
    for p in 0..24 {
        flags.0[p] = if rl >> p & 1 == 1 {
            0xc0
        } else if ud >> p & 1 == 0 {
            0x40
        } else {
            0
        };
    }
    flags
}

/// The staging table's index of the state with U/D mask `ud` (complement `nud`) and R/L mask
/// `rl`: the large table's, or the symmetry-reduced one's.
fn staging_index(t: &P1Tables, ud: u32, nud: u32, rl: u32) -> u64 {
    let st = &t.staging;
    if t.big {
        st.big_index_n(ud, nud, rl)
    } else {
        st.fast_index(unsafe { *st.info(ud) }, rl)
    }
}

/// The masks over `moves` of the moves that decrease and that keep the staging distance of
/// the state with U/D mask `ud` and R/L mask `rl`, from the staging table (either).
fn masks_of(t: &P1Tables, ud: u32, rl: u32, moves: &[V32]) -> (u64, u64) {
    let v = t.table.get(staging_index(t, ud, !ud & MASK24, rl));
    if v == 3 {
        return (0, 0);
    }
    let flags = flags_of(ud, rl);
    let mut idx = [0u64; N36];
    for (e, mv) in idx.iter_mut().zip(moves) {
        let (a, b, c) = flag_masks(&flags.gather(mv));
        *e = staging_index(t, a, b, c);
        t.table.prefetch(*e);
    }
    let (mut dec, mut eq) = (0, 0);
    for (m, &e) in idx[..moves.len()].iter().enumerate() {
        let w = t.table.get(e);
        dec |= ((w == (v + 2) % 3) as u64) << m;
        eq |= ((w == v) as u64) << m;
    }
    (dec, eq)
}

impl MaskTable {
    fn empty(kind: MaskKind) -> MaskTable {
        MaskTable {
            kind,
            batch_leaves: std::env::var_os("EGGG_P1_BATCH_LEAVES").is_some(),
            batch: std::env::var("EGGG_P1_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32),
            slack_batch: std::env::var("EGGG_P1_SLACK_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SLACK_BATCH),
            big: HugeBuf::zeroed(1),
            red: HugeBuf::zeroed(1),
            shift: 0,
            pair: HugeBuf::zeroed(1),
            lazy: false,
            conj: vec![],
            use_succ: std::env::var("EGGG_P1_SUCC").map_or(true, |v| v != "0"),
            conj_move: vec![],
            succ_class: vec![],
            sym_mul: vec![],
        }
    }

    /// A pair "table" whose entries are computed when read, through the pair table's
    /// indexing and conjugation (tests).
    pub fn lazy_pair_table() -> MaskTable {
        MaskTable {
            lazy: true,
            conj: Self::conj_tables(),
            ..Self::empty(MaskKind::Pair)
        }
    }

    /// A mask "table" that computes each mask from the staging table (tests).
    pub fn virtual_table() -> MaskTable {
        Self::empty(MaskKind::Virtual)
    }

    /// Per symmetry, per (searched) move m: pi(m), the move whose centre permutation is
    /// p o C_m o p^-1, i.e. the move m seen on the symmetric image of the state.
    fn conj_moves() -> Vec<[u8; N_MOVES]> {
        let mv = moves36();
        centre_syms()
            .iter()
            .map(|p| {
                std::array::from_fn(|m| {
                    let cm = &mv[m];
                    (0..N36)
                        .find(|&j| (0..24).all(|i| mv[j].0[p[i] as usize] == p[cm.0[i] as usize]))
                        .expect("conjugate move") as u8
                })
            })
            .collect()
    }

    /// The conjugation lookups of the reduced table.
    fn conj_tables() -> Vec<[[u32; 256]; 5]> {
        let mut out = vec![];
        for pi in Self::conj_moves() {
            let mut lut = [[0u32; 256]; 5];
            for (m, &j) in pi.iter().enumerate() {
                let j = j as usize;
                for v in 0..256 {
                    if v >> (j % 8) & 1 == 1 {
                        lut[j / 8][v] |= 1 << m;
                    }
                }
            }
            out.push(lut);
        }
        out
    }

    /// The reduced table's lookups: the conjugations, and those of the successors.
    fn with_lookups(mut self, st: &Staging) -> MaskTable {
        self.conj = Self::conj_tables();
        self.conj_move = Self::conj_moves();
        let mv = moves36();
        self.succ_class = Vec::with_capacity(N_CLASSES_A * N36);
        for &rep in &st.class_rep {
            let flags = flags_of(rep, 0);
            for k in &mv {
                self.succ_class
                    .push(unsafe { *st.info(flag_masks(&flags.gather(k)).0) });
            }
        }
        let syms = centre_syms();
        self.sym_mul = Vec::with_capacity(48 * 48);
        for a in &syms {
            for b in &syms {
                let p: [u8; 24] = std::array::from_fn(|i| a[b[i] as usize]);
                self.sym_mul
                    .push(syms.iter().position(|q| *q == p).expect("product") as u8);
            }
        }
        self
    }

    /// Back the table with huge pages (see `HugeBuf::collapse`).
    pub fn collapse(&self) {
        match self.kind {
            MaskKind::Big => self.big.collapse(),
            MaskKind::Reduced => self.red.collapse(),
            MaskKind::Pair => self.pair.collapse(),
            MaskKind::Virtual => {}
        }
    }

    /// The first index of a batch node (state `s`, search over `pair`): the large staging
    /// index (large table), or the rank of the U/D positions (reduced table) or of the pair's
    /// positions (pair table), whose class info [`MaskTable::resolve`] reads.
    #[inline(always)]
    pub fn pre_index(&self, t: &P1Tables, s: &P1State, pair: usize) -> u64 {
        match self.kind {
            MaskKind::Pair => {
                let g = t.pat.as_ref().unwrap().flags(&s.centres, pair);
                t.staging.rank24(crate::simd::movemask(&g)) as u64
            }
            MaskKind::Reduced => t.staging.rank24(s.masks().0) as u64,
            MaskKind::Big | MaskKind::Virtual => {
                let (ud, rl) = s.masks();
                t.staging.big_index(ud, rl)
            }
        }
    }

    /// [`MaskTable::pre_index`] of a state with large staging index `i` (not the pair table).
    #[inline(always)]
    pub fn pre_index_of_big(&self, i: u64) -> u64 {
        if self.kind == MaskKind::Reduced {
            i / N_SPLIT as u64
        } else {
            i
        }
    }

    /// Prefetch what the mask of the node with first index `i` needs first.
    #[inline(always)]
    pub fn prefetch(&self, t: &P1Tables, i: u64) {
        match self.kind {
            MaskKind::Big => {
                crate::table::prefetch(unsafe { self.big.as_ptr().add(i as usize) });
            }
            MaskKind::Reduced | MaskKind::Pair => crate::table::prefetch(t.staging.info_of_rank(i)),
            MaskKind::Virtual => {}
        }
    }

    /// The reduced (or pair) table's index and symmetry for the node with first index `i`
    /// (its class info prefetched), the entry prefetched.
    #[inline(always)]
    pub fn resolve(&self, t: &P1Tables, i: u64, s: &P1State, pair: usize) -> (u64, u8) {
        let st = &t.staging;
        if self.kind == MaskKind::Pair {
            let pt = t.pat.as_ref().unwrap();
            let info = unsafe { *st.info_of_rank(i) };
            let j = pt.red_index(st, info, &pt.flags(&s.centres, pair));
            crate::table::prefetch(unsafe { self.pair.as_ptr().add(5 * j as usize) });
            return (j, info as u8);
        }
        let info = unsafe { *st.info_of_rank(i) };
        let j = st.fast_index_flags(info, &s.centre_flags());
        self.prefetch_entry(j);
        (j, info as u8)
    }

    /// The reduced table's entry `i` (its first 16 bytes).
    #[inline(always)]
    fn ent(&self, i: u64) -> [u64; 2] {
        self.red[(i as usize) << self.shift]
    }

    /// Prefetch the reduced table's entry `j`.
    #[inline(always)]
    pub fn prefetch_entry(&self, j: u64) {
        crate::table::prefetch(unsafe { self.red.as_ptr().add((j as usize) << self.shift) });
    }

    /// The successors of the node with reduced index `i`, if its entry holds them (reduced
    /// table only; see [`MaskTable::add_succ`]).
    #[inline(always)]
    pub fn succ(&self, i: u64) -> Option<Succ> {
        if !self.use_succ {
            return None;
        }
        let ext = if self.shift == 1 {
            self.red[2 * i as usize + 1]
        } else {
            [0; 2]
        };
        Self::succ_of(self.ent(i), ext, i)
    }

    /// [`MaskTable::succ`] of entry `i` holding `e` (and in the wide table `ext`).
    #[inline(always)]
    fn succ_of(e: [u64; 2], ext: [u64; 2], i: u64) -> Option<Succ> {
        let [w0, w1] = e;
        if w1 >> SUCC_FLAG & 1 == 0 {
            return None;
        }
        let half = (1u64 << (2 * SUCC_BITS)) - 1;
        let eq = w1 >> SUCC_EQ & 1 == 1;
        Some(Succ {
            list: (w0 | if eq { w1 } else { 0 }) & MASK36,
            ch: [
                w0 >> SUCC_SHIFT0 | (w1 >> N36 & half) << (2 * SUCC_BITS),
                ext[0],
                ext[1],
            ],
            row: (i / N_SPLIT as u64) as usize * N36,
            eq,
        })
    }

    /// The reduced index and symmetry of the child after (searched) move `m`, one of the
    /// moves whose successors `sc` holds, of the node with symmetry `sym`: the stored
    /// successor of its representative after the conjugate move k, the child's symmetry
    /// that of the successor's class times the node's.
    #[inline(always)]
    pub fn succ_child(&self, sc: &Succ, sym: u8, m: usize) -> (u64, u8) {
        debug_assert!(m < N_MOVES && (sym as usize) < 48);
        // SAFETY: sym < 48 (a resolved node's), m < N_MOVES, k < 36, and so is the info's
        // symmetry; a slot is below the number of stored successors (at most 12).
        unsafe {
            let k = *self.conj_move.get_unchecked(sym as usize).get_unchecked(m) as u32;
            debug_assert!(sc.list >> k & 1 == 1);
            let slot = (sc.list & ((1u64 << k) - 1)).count_ones();
            let u = sc.ch.get_unchecked(slot as usize / 4) >> (SUCC_BITS * (slot % 4))
                & ((1 << SUCC_BITS) - 1);
            let info = *self.succ_class.get_unchecked(sc.row + k as usize);
            let j = (info >> 8) as u64 * N_SPLIT as u64 + u;
            let t = *self
                .sym_mul
                .get_unchecked((info & 255) as usize * 48 + sym as usize);
            (j, t)
        }
    }

    /// Whether a node at staging distance = moves left (>= 1) with reduced index `i` (as
    /// above) and odd (`odd`) or even wing parity can still reach an endpoint: its entry's
    /// viability bit (see [`MaskTable::add_viable`]; reduced table only).
    #[inline(always)]
    pub fn viable(&self, i: u64, odd: bool) -> bool {
        self.ent(i)[0] >> (VIABLE_SHIFT + odd as u32) & 1 == 1
    }

    /// The 27-move masks of the moves that decrease and that keep the staging distance, of
    /// the state with reduced index `i` and symmetry `sym` (see [`MaskTable::resolve`];
    /// reduced table only).
    #[inline(always)]
    pub fn get_dec_eq(&self, i: u64, sym: u8) -> (u32, u32) {
        let [d, e] = self.ent(i);
        let c = unsafe { self.conj.get_unchecked(sym as usize) };
        let f = |x: u64| {
            c[0][(x & 255) as usize]
                | c[1][(x >> 8 & 255) as usize]
                | c[2][(x >> 16 & 255) as usize]
                | c[3][(x >> 24 & 255) as usize]
                | c[4][(x >> 32 & 255) as usize]
        };
        (f(d), f(e))
    }

    /// The 27-move mask of state `s` (index `i` and symmetry `sym` as above); the virtual
    /// table also serves the per-pair searches (`pair`: the pattern table's distance).
    #[inline(always)]
    pub fn get(&self, t: &P1Tables, i: u64, sym: u8, s: &P1State, pair: usize) -> u32 {
        match self.kind {
            MaskKind::Big => self.big[i as usize],
            MaskKind::Reduced | MaskKind::Pair => {
                let e = if self.lazy {
                    let pt = t.pat.as_ref().unwrap();
                    let g = pt.red_state(&t.staging, i as usize);
                    pair_mask_of(t, &g, pt.table.get(i), &moves36())
                } else if self.kind == MaskKind::Pair {
                    // SAFETY: 5 i + 8 is within the buffer (3 bytes of padding).
                    unsafe {
                        (self.pair.as_ptr().add(5 * i as usize) as *const u64).read_unaligned()
                    }
                } else {
                    self.ent(i)[0]
                };
                // (The conjugation lookups of the fifth byte read only its four mask bits,
                // not the reduced entries' viability bits.)
                let c = unsafe { self.conj.get_unchecked(sym as usize) };
                c[0][(e & 255) as usize]
                    | c[1][(e >> 8 & 255) as usize]
                    | c[2][(e >> 16 & 255) as usize]
                    | c[3][(e >> 24 & 255) as usize]
                    | c[4][(e >> 32 & 255) as usize]
            }
            MaskKind::Virtual => {
                let v = t.value_pair(s, pair);
                let mut mask = 0;
                for m in 0..N_MOVES {
                    if t.value_pair(&s.apply(m, t), pair) == (v + 2) % 3 {
                        mask |= 1 << m;
                    }
                }
                mask
            }
        }
    }

    /// The large table, from the large staging table (`t.big`).
    pub fn generate_big(t: &P1Tables) -> MaskTable {
        assert!(t.big);
        let mut big = HugeBuf::<u32>::zeroed(Staging::big_len());
        let ptr = big.as_mut_slice().as_mut_ptr() as usize;
        let mv = moves36();
        crate::par::for_chunks(735_471, 64, |r| {
            for rank in r {
                let ud = comb_unrank(rank as u32, 24, 8);
                for j in 0..N_SPLIT {
                    let rl = pdep(t.staging.unl_inv[j] as u32, !ud & MASK24);
                    let m = masks_of(t, ud, rl, &mv[..N_MOVES]).0 as u32;
                    // SAFETY: each index is written by one thread only, within the buffer.
                    unsafe { *(ptr as *mut u32).add(rank * N_SPLIT + j) = m };
                }
            }
        });
        MaskTable {
            big,
            ..Self::empty(MaskKind::Big)
        }
    }

    /// The pair table, from the reduced pattern table (`t.pat`).
    pub fn generate_pair(t: &P1Tables) -> MaskTable {
        let pt = t.pat.as_ref().unwrap();
        assert!(!pt.big);
        let n = crate::p1pat::N_RED;
        let mut pair = HugeBuf::<u8>::zeroed(5 * n + 3);
        let ptr = pair.as_mut_slice().as_mut_ptr() as usize;
        let mv = moves36();
        crate::par::for_chunks(n, 1 << 12, |r| {
            for i in r {
                let e = pair_mask_of(t, &pt.red_state(&t.staging, i), pt.table.get(i as u64), &mv);
                // SAFETY: each entry's 5 bytes are written by one thread only.
                unsafe {
                    let p = (ptr as *mut u8).add(5 * i);
                    for k in 0..5 {
                        *p.add(k) = (e >> (8 * k)) as u8;
                    }
                }
            }
        });
        MaskTable {
            pair,
            conj: Self::conj_tables(),
            ..Self::empty(MaskKind::Pair)
        }
    }

    /// Set the viability bits of the reduced table's entries (bits 36 and 37 of each, for
    /// even and odd wing parity), and return how many entries with a non-empty mask have
    /// each bit clear. The bit of a state at staging distance d >= 1 says whether, with that
    /// wing parity, a path of exactly d moves (each decreasing the distance) reaches staged
    /// centres with even wing parity, its last move a wide quarter turn.
    ///
    /// Below a node at distance = moves left, every path to an endpoint is such a path: each
    /// move must decrease the distance, an endpoint has even wing parity, and from unstaged
    /// centres the last move is a wide move (an outer move leaves every face's centres on
    /// that face) and not a half turn (those are phase-2 moves in every frame; see
    /// `Search1::leaf`). So a node whose bit is clear has no endpoint below it. The paths are
    /// over the 36 moves (a superset of the solver's), which the 48 symmetries map onto
    /// themselves, wide quarter turns onto wide quarter turns: the bits are invariant, and the
    /// representative's are the searched state's. Generated from the masks alone by memoised
    /// recursion (a few seconds).
    pub fn add_viable(&mut self, st: &Staging) -> [u64; 2] {
        use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
        struct Ctx<'a> {
            st: &'a Staging,
            red: &'a HugeBuf<[u64; 2]>,
            mv: [V32; N36],
            axes: [u32; 3],
            /// Per entry, per parity (2 bits each): known, viable.
            memo: Vec<AtomicU8>,
        }
        fn viable(c: &Ctx, i: u64, odd: bool) -> bool {
            let sh = 2 * odd as u32;
            let e = c.memo[i as usize].load(Relaxed);
            if e >> sh & 1 == 1 {
                return e >> (sh + 1) & 1 == 1;
            }
            let (ud, rl) = c.st.rep_of(i);
            let flags = flags_of(ud, rl);
            let mut dec = c.red[i as usize][0] & MASK36;
            let mut v = false;
            while dec != 0 && !v {
                let m = dec.trailing_zeros() as usize;
                dec &= dec - 1;
                let quarter = WIDE_QUARTER36 >> m & 1 == 1;
                let (u2, _, r2) = flag_masks(&flags.gather(&c.mv[m]));
                v = if c.axes.contains(&u2) && c.axes.contains(&r2) {
                    // Staged (distance 1): the move must be a wide quarter turn from odd parity.
                    quarter && odd
                } else {
                    viable(c, c.st.index(u2, r2), odd ^ quarter)
                };
            }
            c.memo[i as usize].fetch_or((1 | (v as u8) << 1) << sh, Relaxed);
            v
        }
        let n = N_CLASSES_A * N_SPLIT;
        let c = Ctx {
            st,
            red: &self.red,
            mv: moves36(),
            axes: Staging::axis_masks(),
            memo: (0..n).map(|_| AtomicU8::new(0)).collect(),
        };
        crate::par::for_chunks(n, 1 << 12, |r| {
            for i in r {
                viable(&c, i as u64, false);
                viable(&c, i as u64, true);
            }
        });
        let memo = c.memo;
        let mut dead = [0u64; 2];
        for (i, e) in memo.into_iter().enumerate() {
            let e = e.into_inner();
            let bits = (e >> 1 & 1) as u64 | ((e >> 3 & 1) as u64) << 1;
            let e = &mut self.red[i][0];
            if *e & MASK36 != 0 {
                dead[0] += (bits & 1 == 0) as u64;
                dead[1] += (bits & 2 == 0) as u64;
            }
            *e = *e & MASK36 | bits << VIABLE_SHIFT;
        }
        dead
    }

    /// The reduced table (without the viability bits), from the staging table (either).
    pub fn generate_reduced(t: &P1Tables) -> MaskTable {
        let n = N_CLASSES_A * N_SPLIT;
        let mut red = HugeBuf::<[u64; 2]>::zeroed(n);
        let ptr = red.as_mut_slice().as_mut_ptr() as usize;
        let mv = moves36();
        crate::par::for_chunks(n, 1 << 12, |r| {
            for i in r {
                let (ud, rl) = t.staging.rep_of(i as u64);
                let (dec, eq) = masks_of(t, ud, rl, &mv);
                // SAFETY: each index is written by one thread only, within the buffer.
                unsafe { *(ptr as *mut [u64; 2]).add(i) = [dec, eq] };
            }
        });
        MaskTable {
            red,
            ..Self::empty(MaskKind::Reduced)
        }
        .with_lookups(&t.staging)
    }

    /// Store the successors of the reduced table's entries in their spare bits (after
    /// [`MaskTable::add_viable`]): those of the decreasing and equal moves (of the 36) where
    /// there are at most [`SUCC_MAX`] of them, else of the decreasing moves where there are
    /// at most [`SUCC_MAX`]. Returns the number of entries with successors.
    ///
    /// The successor after move k of an entry is the reduced index of its representative's
    /// child after k: that child's class (and the symmetry that takes its U/D mask to the
    /// class representative) depends only on the entry's class and k (see `succ_class`), so
    /// the entry stores only the child's split of the other 16 positions (13 bits; in the
    /// order of the moves). In the search, the child after move m of a node with symmetry s
    /// (which takes the node to its entry's representative, up to swapping the R/L and F/B
    /// colours, which leaves the masks unchanged) is the image under s^-1 of the
    /// representative's child after the conjugate move pi_s(m), so it takes the successor's
    /// index and the symmetry s' s, s' the successor's class symmetry. Where the successor's
    /// class has a non-trivial stabiliser this may be another stabiliser variant of the index
    /// that resolving the child computes; both entries hold the same masks under their
    /// symmetries. Deep in the search, over 95% of the nodes at distance = moves left have at
    /// most 4 decreasing moves (among the 36, 2.1 on average): their children need no class
    /// info, index or symmetry computation.
    pub fn add_succ(&mut self, st: &Staging) -> u64 {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        assert_eq!(self.shift, 0);
        let n = N_CLASSES_A * N_SPLIT;
        let ptr = self.red.as_mut_slice().as_mut_ptr() as usize;
        let mv = moves36();
        let count = AtomicU64::new(0);
        crate::par::for_chunks(n, 1 << 12, |r| {
            let mut local = 0;
            for i in r {
                // SAFETY: each entry is written by one thread only, within the buffer.
                let e = unsafe { &mut *(ptr as *mut [u64; 2]).add(i) };
                *e = with_succ(st, &mv, i as u64, *e, SUCC_MAX)[0];
                local += e[1] >> SUCC_FLAG & 1;
            }
            count.fetch_add(local, Relaxed);
        });
        count.into_inner()
    }

    /// The wide table (32-byte entries: up to [`SUCC_MAX_WIDE`] successors, see
    /// [`MaskTable::add_succ`]) from the reduced one (its masks and viability bits), and the
    /// number of entries with successors.
    pub fn widen(&self, st: &Staging) -> (MaskTable, u64) {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        assert_eq!((self.kind, self.shift), (MaskKind::Reduced, 0));
        let n = N_CLASSES_A * N_SPLIT;
        let mut red = HugeBuf::<[u64; 2]>::zeroed(2 * n);
        let ptr = red.as_mut_slice().as_mut_ptr() as usize;
        let mv = moves36();
        let count = AtomicU64::new(0);
        crate::par::for_chunks(n, 1 << 12, |r| {
            let mut local = 0;
            for i in r {
                let w = with_succ(st, &mv, i as u64, self.red[i], SUCC_MAX_WIDE);
                local += w[0][1] >> SUCC_FLAG & 1;
                // SAFETY: each entry is written by one thread only, within the buffer.
                unsafe { (ptr as *mut [[u64; 2]; 2]).add(i).write(w) };
            }
            count.fetch_add(local, Relaxed);
        });
        let m = MaskTable {
            red,
            shift: 1,
            ..Self::empty(MaskKind::Reduced)
        };
        (m.with_lookups(st), count.into_inner())
    }
}

/// Reduced entry `i` (`e`: its masks and viability bits; any successors in it are ignored)
/// with at most `max` successors (see [`MaskTable::add_succ`]), and the further successors
/// of a wide entry (`max` > 4).
fn with_succ(st: &Staging, mv: &[V32; N36], i: u64, e: [u64; 2], max: u32) -> [[u64; 2]; 2] {
    let (dec, eq) = (e[0] & MASK36, e[1] & MASK36);
    let e = [e[0] & ((1 << SUCC_SHIFT0) - 1), eq];
    let list = if (dec | eq).count_ones() <= max {
        dec | eq
    } else {
        dec
    };
    if dec == 0 || list.count_ones() > max {
        return [e, [0; 2]];
    }
    let (ud, rl) = st.rep_of(i);
    let flags = flags_of(ud, rl);
    let (mut ch, mut slot, mut d) = ([0u64; 3], 0, list);
    while d != 0 {
        let k = d.trailing_zeros() as usize;
        d &= d - 1;
        let g = flags.gather(&mv[k]);
        let info = unsafe { *st.info(flag_masks(&g).0) };
        let j = st.fast_index_flags(info, &g);
        debug_assert_eq!(j as usize / N_SPLIT, (info >> 8) as usize);
        ch[slot / 4] |= (j % N_SPLIT as u64) << (SUCC_BITS as usize * (slot % 4));
        slot += 1;
    }
    let half = (1u64 << (2 * SUCC_BITS)) - 1;
    [
        [
            e[0] | (ch[0] & half) << SUCC_SHIFT0,
            eq | (ch[0] >> (2 * SUCC_BITS)) << N36
                | 1 << SUCC_FLAG
                | ((list != dec) as u64) << SUCC_EQ,
        ],
        [ch[1], ch[2]],
    ]
}

/// Which move-mask table to load (see [`load_sel`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaskSel {
    /// The reduced table (1.6 GB, with up to 4 successors per entry).
    Reduced,
    /// The wide reduced table (3.2 GB, with up to 12 successors per entry).
    Wide,
    /// The large table (18.9 GB, no symmetry reduction).
    Big,
}

/// [`load_sel`] of the large (`big`) or the reduced table.
pub fn load(dir: &std::path::Path, t: &P1Tables, big: bool, verbose: bool) -> MaskTable {
    load_sel(
        dir,
        t,
        if big { MaskSel::Big } else { MaskSel::Reduced },
        verbose,
    )
}

/// Load (or generate, and save) a mask table in `dir`: the large one (18.9 GB,
/// `p1-mask-big.tab`; from the large staging table), the reduced one (1.6 GB, `p1-mask3.tab`,
/// with successors; from `p1-mask2.tab`, the same without them, if present, else from
/// either staging table) or the wide one (3.2 GB, `p1-succ.tab`; from the reduced one).
/// `EGGG_P1_MASK_VIRTUAL` set: no table, masks computed when needed (slow; tests).
pub fn load_sel(dir: &std::path::Path, t: &P1Tables, sel: MaskSel, verbose: bool) -> MaskTable {
    if std::env::var_os("EGGG_P1_MASK_VIRTUAL").is_some() {
        return MaskTable::virtual_table();
    }
    if t.pat.is_some() {
        return load_pair(dir, t, verbose);
    }
    match sel {
        MaskSel::Big => {
            let name = "p1-mask-big";
            let path = dir.join(format!("{name}.tab"));
            let mut m = MaskTable::empty(MaskKind::Big);
            m.big = HugeBuf::zeroed(Staging::big_len());
            if crate::table::load(&path, name, m.big.as_bytes_mut()).is_ok() {
                return m;
            }
            drop(m);
            if verbose {
                eprintln!("generating the phase-1 move-mask table (18.9 GB)...");
            }
            let m = MaskTable::generate_big(t);
            crate::table::save(&path, name, m.big.as_bytes()).expect("saving table");
            m
        }
        MaskSel::Reduced => load_reduced(dir, t, verbose),
        MaskSel::Wide => {
            let name = "p1-succ";
            let path = dir.join(format!("{name}.tab"));
            let mut m = MaskTable::empty(MaskKind::Reduced);
            m.red = HugeBuf::zeroed(2 * N_CLASSES_A * N_SPLIT);
            m.shift = 1;
            if crate::table::load(&path, name, m.red.as_bytes_mut()).is_ok() {
                return m.with_lookups(&t.staging);
            }
            drop(m);
            let r = load_reduced(dir, t, verbose);
            if verbose {
                eprintln!("generating the wide phase-1 move-mask table (3.2 GB)...");
            }
            let t0 = std::time::Instant::now();
            let (m, n) = r.widen(&t.staging);
            drop(r);
            if verbose {
                eprintln!(
                    "wide phase-1 mask successors in {:.1} s ({n} of {} entries)",
                    t0.elapsed().as_secs_f64(),
                    N_CLASSES_A * N_SPLIT
                );
            }
            crate::table::save(&path, name, m.red.as_bytes()).expect("saving table");
            m
        }
    }
}

/// [`load_sel`] of the reduced table.
fn load_reduced(dir: &std::path::Path, t: &P1Tables, verbose: bool) -> MaskTable {
    // (The reduced tables of earlier versions: "p1-mask" without the equal masks,
    // "p1-mask2" without the successors.)
    let name = "p1-mask3";
    let path = dir.join(format!("{name}.tab"));
    let mut m = MaskTable::empty(MaskKind::Reduced);
    m.red = HugeBuf::zeroed(N_CLASSES_A * N_SPLIT);
    if crate::table::load(&path, name, m.red.as_bytes_mut()).is_ok() {
        return m.with_lookups(&t.staging);
    }
    let mut m = if crate::table::load(&dir.join("p1-mask2.tab"), "p1-mask2", m.red.as_bytes_mut())
        .is_ok()
    {
        m.with_lookups(&t.staging)
    } else {
        drop(m);
        if verbose {
            eprintln!("generating the phase-1 move-mask table (1.6 GB)...");
        }
        let mut m = MaskTable::generate_reduced(t);
        let t0 = std::time::Instant::now();
        let dead = m.add_viable(&t.staging);
        if verbose {
            eprintln!(
                "phase-1 mask viability bits in {:.1} s ({} / {} of {} entries dead for even \
                 / odd wing parity)",
                t0.elapsed().as_secs_f64(),
                dead[0],
                dead[1],
                m.red.len()
            );
        }
        m
    };
    let t0 = std::time::Instant::now();
    let n = m.add_succ(&t.staging);
    if verbose {
        eprintln!(
            "phase-1 mask successors in {:.1} s ({n} of {} entries)",
            t0.elapsed().as_secs_f64(),
            m.red.len()
        );
    }
    crate::table::save(&path, name, m.red.as_bytes()).expect("saving table");
    m
}

/// The mask over the 36 moves of pattern-distance-decreasing moves of the state with pair
/// flags `g` (see `PatTable::flags`) and stored value `v`, from the reduced pattern table.
fn pair_mask_of(t: &P1Tables, g: &V32, v: u8, moves: &[V32; N36]) -> u64 {
    let pt = t.pat.as_ref().unwrap();
    let st = &t.staging;
    if v == 3 {
        return 0;
    }
    let mut idx = [0u64; N36];
    for (e, mv) in idx.iter_mut().zip(moves) {
        let h = g.gather(mv);
        let info = unsafe { *st.info(crate::simd::movemask(&h)) };
        *e = pt.red_index(st, info, &h);
        pt.table.prefetch(*e);
    }
    let want = (v + 2) % 3;
    let mut mask = 0;
    for (m, &e) in idx.iter().enumerate() {
        if pt.table.get(e) == want {
            mask |= 1 << m;
        }
    }
    mask
}

/// Load (or generate from the reduced pattern table, ~17.5 GB, and save) the pair table.
fn load_pair(dir: &std::path::Path, t: &P1Tables, verbose: bool) -> MaskTable {
    let name = "p1-pair-mask";
    let path = dir.join(format!("{name}.tab"));
    let mut m = MaskTable::empty(MaskKind::Pair);
    m.pair = HugeBuf::zeroed(5 * crate::p1pat::N_RED + 3);
    if crate::table::load(&path, name, m.pair.as_bytes_mut()).is_ok() {
        m.conj = MaskTable::conj_tables();
        return m;
    }
    drop(m);
    if verbose {
        eprintln!("generating the phase-1 pair move-mask table (17.5 GB)...");
    }
    let m = MaskTable::generate_pair(t);
    crate::table::save(&path, name, m.pair.as_bytes()).expect("saving table");
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::Cube;

    /// The large and reduced tables' masks (as computed by their generators, from either
    /// staging table, for random states) are the moves that decrease and keep the distance
    /// (needs the large staging table in ./tables; skipped without it).
    #[test]
    fn masks_are_decreasing_moves() {
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-staging-big.tab").exists() {
            return;
        }
        let t = crate::solver::load_p1(dir, true, 8, false);
        let small = crate::solver::load_p1(dir, false, 8, false);
        let v = MaskTable::virtual_table();
        let mv = moves36();
        let conj = MaskTable::conj_tables();
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let mut s = P1State::from_cube(&Cube::random(&mut rng));
            let mut d = t.distance(&s);
            for k in 0..12 {
                let want = v.get(&t, 0, 0, &s, crate::phase1::ALL_PAIRS);
                let want_eq = (0..N_MOVES)
                    .filter(|&m| t.distance(&s.apply(m, &t)) == d)
                    .fold(0, |a, m| a | 1 << m);
                let (ud, rl) = s.masks();
                let (dec, eq) = masks_of(&t, ud, rl, &mv[..N_MOVES]);
                assert_eq!((dec as u32, eq as u32), (want, want_eq));
                // The reduced table's entry: the 36-move masks of the symmetric image (from
                // either staging table).
                let info = unsafe { *t.staging.info(ud) };
                let j = t.staging.fast_index_flags(info, &s.centre_flags());
                let (ru, rr) = t.staging.rep_of(j);
                let e = masks_of(&t, ru, rr, &mv);
                assert_eq!(masks_of(&small, ru, rr, &mv), e);
                let c = &conj[(info & 255) as usize];
                let f = |x: u64| (0..5).fold(0, |a, k| a | c[k][(x >> (8 * k) & 255) as usize]);
                assert_eq!((f(e.0), f(e.1)), (want, want_eq));
                let m = (k * 5 + 1) % N_MOVES;
                s = s.apply(m, &t);
                d = t.distance(&s);
            }
        }
    }

    /// The reduced table's entries, as the search reads them (by the searched state's index
    /// and symmetry), give the moves that decrease and keep the staging distance, by the
    /// symmetry-reduced staging table (needs `p1-mask2.tab` in ./tables, made by
    /// `eggg-solver gen --tier large`; skipped without it).
    #[test]
    fn reduced_entries_give_child_distances() {
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-mask2.tab").exists() {
            return;
        }
        let t = crate::solver::load_p1(dir, false, 8, false);
        let mt = load(dir, &t, false, false);
        let mut rng = rand::thread_rng();
        for _ in 0..300 {
            let mut s = P1State::from_cube(&Cube::random(&mut rng));
            for k in 0..12 {
                let d = t.distance(&s);
                let (mut dec, mut eq) = (0, 0);
                for m in 0..N_MOVES {
                    let dc = t.distance(&s.apply(m, &t));
                    dec |= ((dc + 1 == d) as u32) << m;
                    eq |= ((dc == d) as u32) << m;
                }
                let i = mt.pre_index(&t, &s, crate::phase1::ALL_PAIRS);
                let (j, sym) = mt.resolve(&t, i, &s, crate::phase1::ALL_PAIRS);
                assert_eq!(mt.get_dec_eq(j, sym), (dec, eq));
                assert_eq!(mt.get(&t, j, sym, &s, crate::phase1::ALL_PAIRS), dec);
                s = s.apply((k * 7 + 2) % N_MOVES, &t);
            }
        }
    }

    /// The successors, as the search reads them (by the node's reduced index and symmetry,
    /// through the conjugate move, the class lookup and the symmetry product), are the
    /// children: the child mapped by the successor's symmetry is the successor's
    /// representative (up to swapping the R/L and F/B colours), the successor's masks under
    /// that symmetry are the child's, and resolving the child gives the same index or another
    /// stabiliser variant of it. Checked for every move (an entry made with that move as its
    /// only "decreasing" one) and for the entries' own decreasing moves, from random states,
    /// states near staged centres, states whose U/D mask's class has a non-trivial stabiliser
    /// (under random symmetries) and the parents of such states (needs the symmetry-reduced
    /// staging table in ./tables; no mask table).
    #[test]
    fn successors_are_the_children() {
        use rand::{Rng, SeedableRng, seq::SliceRandom};
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-staging2.tab").exists() {
            return;
        }
        let t = crate::solver::load_p1(dir, false, 8, false);
        let st = &t.staging;
        let mt = MaskTable::empty(MaskKind::Reduced).with_lookups(st);
        let mv = moves36();
        let syms = centre_syms();
        let masks = |j: u64| {
            let (ud, rl) = st.rep_of(j);
            let (d, e) = masks_of(&t, ud, rl, &mv);
            [d, e]
        };
        let direct = |s: &P1State| {
            let (ud, rl) = s.masks();
            let (d, e) = masks_of(&t, ud, rl, &mv[..N_MOVES]);
            (d as u32, e as u32)
        };
        let conj = |e: u64, sym: u8| -> u32 {
            let c = &mt.conj[sym as usize];
            (0..5).fold(0, |a, k| a | c[k][(e >> (8 * k) & 255) as usize])
        };
        let resolve = |s: &P1State| {
            let info = unsafe { *st.info(s.masks().0) };
            (st.fast_index_flags(info, &s.centre_flags()), info as u8)
        };
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(11);
        let mut states = vec![];
        for k in 0..400 {
            let mut s = P1State::from_cube(&Cube::random(&mut rng));
            if k % 2 == 1 {
                s = P1State::from_cube(&Cube::SOLVED);
                for _ in 0..1 + k % 7 {
                    s = s.apply(rng.gen_range(0..N_MOVES), &t);
                }
            }
            states.push(s);
        }
        let stab: Vec<usize> = (0..N_CLASSES_A)
            .filter(|&c| !st.stab[c].is_empty())
            .collect();
        assert!(!stab.is_empty());
        for _ in 0..800 {
            let rep = st.class_rep[stab[rng.gen_range(0..stab.len())]];
            let mut rest: Vec<usize> = (0..24).filter(|&i| rep >> i & 1 == 0).collect();
            rest.shuffle(&mut rng);
            let p = &syms[rng.gen_range(0..48)];
            let mut s = P1State::from_cube(&Cube::SOLVED);
            for i in 0..24 {
                let pair = if rep >> i & 1 == 1 {
                    0
                } else if rest[..8].contains(&i) {
                    1
                } else {
                    2
                };
                s.centres.0[p[i] as usize] = 2 * pair + rng.gen_range(0..2);
            }
            states.push(s);
            // A parent of it.
            states.push(s.apply(rng.gen_range(0..N_MOVES), &t));
        }
        // Children checked: the resolved index, another stabiliser variant; entries' own
        // successors (of their decreasing moves, of their decreasing and equal moves).
        let mut seen = [0usize; 4];
        for s in &states {
            let (j, sym) = resolve(s);
            let e = masks(j);
            assert_eq!((conj(e[0], sym), conj(e[1], sym)), direct(s));
            // The entry's own successors (reduced and wide entries): of the decreasing and
            // equal moves if there are at most `max`, else of the decreasing moves if there
            // are at most `max`.
            let (dec, eq) = (e[0] & MASK36, e[1] & MASK36);
            let full = [SUCC_MAX, SUCC_MAX_WIDE].map(|max| {
                let w = with_succ(st, &mv, j, e, max);
                assert_eq!([w[0][0] & ((1 << SUCC_SHIFT0) - 1), w[0][1] & MASK36], e);
                let sc = MaskTable::succ_of(w[0], w[1], j);
                let n = (dec | eq).count_ones();
                assert_eq!(
                    sc.map(|sc| sc.eq),
                    (dec != 0)
                        .then_some(n <= max)
                        .filter(|&b| b || dec.count_ones() <= max)
                );
                if max == SUCC_MAX {
                    assert_eq!(w[1], [0; 2]);
                }
                sc
            });
            for m in 0..N_MOVES {
                let k = mt.conj_move[sym as usize][m];
                let one = with_succ(st, &mv, j, [1 << k, 0], SUCC_MAX);
                let sc = MaskTable::succ_of(one[0], one[1], j).unwrap();
                let (j2, sym2) = mt.succ_child(&sc, sym, m);
                for sf in full.iter().flatten() {
                    if sf.list >> k & 1 == 1 {
                        assert_eq!(mt.succ_child(sf, sym, m), (j2, sym2));
                        seen[2 + sf.eq as usize] += 1;
                    }
                }
                let c = s.apply(m, &t);
                let (cu, cr) = c.masks();
                let p = &syms[sym2 as usize];
                let img = |x: u32| {
                    (0..24)
                        .filter(|&i| x >> i & 1 == 1)
                        .fold(0, |a, i| a | 1 << p[i])
                };
                let (ru, rr) = st.rep_of(j2);
                assert_eq!(img(cu), ru);
                assert!(img(cr) == rr || img(!(cu | cr) & MASK24) == rr);
                let e2 = masks(j2);
                assert_eq!((conj(e2[0], sym2), conj(e2[1], sym2)), direct(&c));
                let (j3, _) = resolve(&c);
                if j3 == j2 {
                    seen[0] += 1;
                } else {
                    assert_eq!(j3 as usize / N_SPLIT, j2 as usize / N_SPLIT);
                    assert!(!st.stab[j2 as usize / N_SPLIT].is_empty());
                    seen[1] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&n| n > 100), "{seen:?}");
    }

    /// Chains of successors through the reduced table and the wide one, as the search
    /// follows them (each child's index and symmetry from its parent's successors, over
    /// several moves, each a decreasing or, where stored, an equal move), give the same masks
    /// and viability bits as resolving each state (needs `p1-mask3.tab` or `p1-mask2.tab`
    /// in ./tables, made by `eggg-solver gen --tier large`, skipped without; the wide table
    /// only if `p1-succ.tab` is there too).
    #[test]
    fn successor_chains_match_resolved_entries() {
        use rand::{Rng, SeedableRng};
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-mask3.tab").exists() && !dir.join("p1-mask2.tab").exists() {
            return;
        }
        let t = crate::solver::load_p1(dir, false, 8, false);
        let all = crate::phase1::ALL_PAIRS;
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
        for sel in [MaskSel::Reduced, MaskSel::Wide] {
            if sel == MaskSel::Wide && !dir.join("p1-succ.tab").exists() {
                continue;
            }
            let mt = load_sel(dir, &t, sel, false);
            // Steps through successors (of decreasing, of equal moves), through resolving.
            let mut steps = [0usize; 3];
            for _ in 0..2000 {
                let mut s = P1State::from_cube(&Cube::random(&mut rng));
                let (mut j, mut sym) = mt.resolve(&t, mt.pre_index(&t, &s, all), &s, all);
                for _ in 0..16 {
                    let (j3, sym3) = mt.resolve(&t, mt.pre_index(&t, &s, all), &s, all);
                    let (dec, eq) = mt.get_dec_eq(j, sym);
                    assert_eq!((dec, eq), mt.get_dec_eq(j3, sym3));
                    for odd in [false, true] {
                        assert_eq!(mt.viable(j, odd), mt.viable(j3, odd));
                    }
                    let list = mt.succ(j).map(|sc| if sc.eq { dec | eq } else { dec });
                    if let Some(sc) = mt.succ(j)
                        && list != Some(0)
                    {
                        let list = list.unwrap();
                        let ms: Vec<usize> = (0..N_MOVES).filter(|&m| list >> m & 1 == 1).collect();
                        let m = ms[rng.gen_range(0..ms.len())];
                        (j, sym) = mt.succ_child(&sc, sym, m);
                        s = s.apply(m, &t);
                        steps[(eq >> m & 1) as usize] += 1;
                    } else {
                        s = s.apply(rng.gen_range(0..N_MOVES), &t);
                        (j, sym) = mt.resolve(&t, mt.pre_index(&t, &s, all), &s, all);
                        steps[2] += 1;
                    }
                }
            }
            assert!(steps.iter().all(|&n| n > 100), "{sel:?} {steps:?}");
        }
    }

    /// The wide quarter turns among the 36 moves are exactly the moves with an odd wing
    /// permutation.
    #[test]
    fn wide_quarter_turns_flip_the_wing_parity() {
        for m in 0..N36 {
            let face = [crate::cube::U, crate::cube::R, crate::cube::F, D, L, B][m / 3 % 6];
            let width = if m < 18 { 1 } else { 2 };
            let p = Perm::of_turn(Turn::face(face, width, (m % 3) as u8 + 1));
            assert_eq!(p.centre_src, moves36()[m].0[..24]);
            assert_eq!(Cube::parity(&p.wing_src) == 1, WIDE_QUARTER36 >> m & 1 == 1);
        }
    }

    /// The viability bits, as the search reads them (by the searched state's reduced index),
    /// agree with a brute-force search for their paths over the 36 moves, guided by the large
    /// staging table, from random states and from states a few moves from staged centres
    /// (needs the large staging table in ./tables, skipped without it; the reduced table is
    /// loaded, or made and saved).
    #[test]
    fn viable_bits_match_brute_force() {
        use std::collections::HashMap;
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-staging-big.tab").exists() {
            return;
        }
        let t = crate::solver::load_p1(dir, true, 8, false);
        let mt = load(dir, &t, false, false);
        let mv = moves36();
        type Memo = HashMap<(u64, bool), bool>;
        // Is there a path of d distance-decreasing moves to staged centres with even wing
        // parity, its last move a wide quarter turn (state: U/D and R/L masks, distance d >= 1,
        // wing parity odd)?
        fn brute(
            t: &P1Tables,
            mv: &[V32; N36],
            ud: u32,
            rl: u32,
            d: u8,
            odd: bool,
            memo: &mut Memo,
        ) -> bool {
            let key = (t.staging.big_index(ud, rl), odd);
            if let Some(&v) = memo.get(&key) {
                return v;
            }
            let flags = flags_of(ud, rl);
            let mut v = false;
            for (m, g) in mv.iter().enumerate() {
                let (u2, _, r2) = flag_masks(&flags.gather(g));
                let dc =
                    crate::table::Mod3Table::decode(d, t.table.get(t.staging.big_index(u2, r2)));
                if dc + 1 != d {
                    continue;
                }
                let quarter = WIDE_QUARTER36 >> m & 1 == 1;
                v = if dc == 0 {
                    quarter && odd
                } else {
                    brute(t, mv, u2, r2, dc, odd ^ quarter, memo)
                };
                if v {
                    break;
                }
            }
            memo.insert(key, v);
            v
        }
        let mut rng = rand::thread_rng();
        let mut memo = Memo::new();
        let mut seen = [[0u32; 2]; 2];
        for k in 0..600 {
            let mut s = P1State::from_cube(&Cube::SOLVED);
            if k % 3 == 0 {
                s = P1State::from_cube(&Cube::random(&mut rng));
            } else {
                for _ in 0..1 + k % 7 {
                    s = s.apply(rand::Rng::gen_range(&mut rng, 0..N_MOVES), &t);
                }
            }
            let d = t.distance(&s);
            if d == 0 {
                continue;
            }
            let (ud, rl) = s.masks();
            let info = unsafe { *t.staging.info(ud) };
            let j = t.staging.fast_index_flags(info, &s.centre_flags());
            for odd in [false, true] {
                let want = brute(&t, &mv, ud, rl, d, odd, &mut memo);
                assert_eq!(mt.viable(j, odd), want, "distance {d}, odd parity {odd}");
                seen[odd as usize][want as usize] += 1;
            }
        }
        // Both parities, both answers.
        assert!(seen.iter().flatten().all(|&n| n > 0), "{seen:?}");
    }

    /// The pair table's masks (through its indexing and conjugation, entries computed by the
    /// generator's function) are the pattern-distance-decreasing moves (needs the large
    /// staging and reduced pattern tables in ./tables; skipped without them).
    #[test]
    fn pair_masks_are_decreasing_moves() {
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-staging-big.tab").exists() || !dir.join("p1-pat.tab").exists() {
            return;
        }
        let mut t = crate::solver::load_p1(dir, true, 8, false);
        crate::solver::load_pat(&mut t, dir, false, 8, false);
        let v = MaskTable::virtual_table();
        let lazy = MaskTable::lazy_pair_table();
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let mut s = P1State::from_cube(&Cube::random(&mut rng));
            for k in 0..12 {
                for pair in 0..3 {
                    let i = lazy.pre_index(&t, &s, pair);
                    let (j, sym) = lazy.resolve(&t, i, &s, pair);
                    assert_eq!(lazy.get(&t, j, sym, &s, pair), v.get(&t, 0, 0, &s, pair));
                }
                s = s.apply((k * 5 + 1) % N_MOVES, &t);
            }
        }
    }
}
