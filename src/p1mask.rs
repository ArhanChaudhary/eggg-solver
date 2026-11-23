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
//! symmetry that takes its U/D mask to the class representative), with the masks over the
//! representative's 27 moves: the searched state's mask is the stored mask with each move
//! conjugated by that symmetry (four byte lookups). A symmetry can take a searched move to
//! Dw, Lw or Bw, but each of those acts on the centres as Uw, Rw or Fw (same power) followed
//! by a rotation about its axis, which keeps everything an entry records: their bits are those
//! of Uw, Rw and Fw ([`fold`]). Its entries (16 bytes, one cache line; 1.6 GB) hold the masks
//! of the moves that decrease and that keep the distance, and two parity-viability bits (see
//! [`MaskTable::add_viable`]): a node at distance = moves left whose wing parity cannot end
//! at an endpoint is dropped as soon as its entry is read, and a node with slack (distance <
//! moves left) has its children's exact distances from its own entry, one read instead of one
//! per child. They also hold, per move, whether the child via the move can still reach an
//! endpoint if it is at distance = moves left (its viability bit, and whether it has a
//! decreasing move the canonical order allows after the move; see [`MaskTable::add_live`]),
//! so that such children are not queued at all. The search then needs no staging table but
//! for the few nodes above the job prefixes (the symmetry-reduced one, 25 MB, does).
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

/// See [`MaskTable::chunk`].
pub const CHUNK: usize = 4096;
/// The 36 outer block moves: the solver's 27, then Dw, Lw, Bw (each power).
const N36: usize = 36;
/// A reduced entry (`p1-mask-live.tab`, two words). Word 0: the moves that decrease the distance
/// (bits 0..27, the representative's moves), those that keep it (27..54), the two viability
/// bits (54 and 55, see [`MaskTable::add_viable`]), live bits 0..8 (56..64). Word 1: the child
/// bits for a node with even (0..27) and with odd (27..54) wing parity, live bits 8..18
/// (54..64). See [`MaskTable::add_live`] for the child and live bits.
const MASK27: u64 = (1 << N_MOVES) - 1;
const EQ_SHIFT: u32 = 27;
const VIABLE_SHIFT: u32 = 54;
const ODD_SHIFT: u32 = 27;
const LIVE0_SHIFT: u32 = 56;
const LIVE1_SHIFT: u32 = 54;
/// The outer moves (U R F D L B and their powers), the first 18 of the 27.
const N_OUTER: usize = 18;
#[cfg(test)]
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
    /// 27-bit masks, the viability, child and live bits by the reduced staging index (1.6 GB).
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
    /// Batch rounds run in chunks of this many nodes (`EGGG_P1_CHUNK`, default
    /// [`CHUNK`]; see `Search1::run_batch_rounds`).
    pub chunk: usize,
    /// With the reduced table, nodes with slack (distance < moves left) and at most this
    /// many moves left are expanded in batches too (`EGGG_P1_SLACK_BATCH`, default
    /// [`SLACK_BATCH`]; 0: none).
    pub slack_batch: u8,
    big: HugeBuf<u32>,
    /// The reduced table's entries: the decreasing moves' 36-bit mask with the viability bits
    /// above it, and the equal moves' mask (16 bytes, aligned: one cache line).
    red: HugeBuf<[u64; 2]>,
    /// The pair table: 5 bytes per entry (and 3 bytes of padding).
    pair: HugeBuf<u8>,
    /// The pair table's entries are computed when read, from the pattern table (tests).
    lazy: bool,
    /// The pair table: per symmetry, per byte of a stored 36-bit mask, the searched state's
    /// 27-bit mask.
    conj: Vec<[[u32; 256]; 5]>,
    /// The reduced table: per symmetry, per byte of a stored 27-bit mask, the searched
    /// state's mask.
    conj27: Vec<[[u32; 256]; 4]>,
    /// The reduced table: per symmetry, per searched move, its stored move (the
    /// representative's move it is conjugated to, folded: see [`fold`]).
    inv27: Vec<[u8; 32]>,
}

/// The stored move (of the representative's 27) of a move of the 36: Dw, Lw and Bw act on the
/// centres as Uw, Rw and Fw with the same power followed by a rotation about their axis (Dw =
/// Uw y', and so on), which keeps the staging distance, the wing parity and the axes, so
/// their entries' bits are those of Uw, Rw and Fw.
const fn fold(j: usize) -> usize {
    if j >= N_MOVES { j - 9 } else { j }
}

/// Per move, the moves on its axis on later faces (in the order U R F D L B Uw Rw Fw): those
/// the canonical order allows after it on its axis.
const LATER_SAME_AXIS: [u32; N_MOVES] = {
    let mut t = [0u32; N_MOVES];
    let mut m = 0;
    while m < N_MOVES {
        let mut y = 0;
        while y < N_MOVES {
            if (y / 3) % 3 == (m / 3) % 3 && y / 3 > m / 3 {
                t[m] |= 1 << y;
            }
            y += 1;
        }
        m += 1;
    }
    t
};

/// Per move, the moves on other axes.
const OTHER_AXES: [u32; N_MOVES] = {
    let mut t = [0u32; N_MOVES];
    let mut m = 0;
    while m < N_MOVES {
        let mut y = 0;
        while y < N_MOVES {
            if (y / 3) % 3 != (m / 3) % 3 {
                t[m] |= 1 << y;
            }
            y += 1;
        }
        m += 1;
    }
    t
};

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
            chunk: std::env::var("EGGG_P1_CHUNK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(CHUNK)
                .max(1),
            slack_batch: std::env::var("EGGG_P1_SLACK_BATCH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(SLACK_BATCH),
            big: HugeBuf::zeroed(1),
            red: HugeBuf::zeroed(1),
            pair: HugeBuf::zeroed(1),
            lazy: false,
            conj: vec![],
            conj27: vec![],
            inv27: vec![],
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

    /// Per symmetry, per searched move m: pi(m), the move (of the 36) whose centre
    /// permutation is p o C_m o p^-1, i.e. the move m seen on the symmetric image of the state.
    fn conj_moves() -> Vec<[usize; N_MOVES]> {
        let mv = moves36();
        centre_syms()
            .iter()
            .map(|p| {
                std::array::from_fn(|m| {
                    (0..N36)
                        .find(|&j| {
                            (0..24).all(|i| mv[j].0[p[i] as usize] == p[mv[m].0[i] as usize])
                        })
                        .expect("conjugate move")
                })
            })
            .collect()
    }

    /// The conjugation lookups of the pair table (36-move masks).
    fn conj_tables() -> Vec<[[u32; 256]; 5]> {
        Self::conj_moves()
            .iter()
            .map(|pi| {
                let mut lut = [[0u32; 256]; 5];
                for (m, &j) in pi.iter().enumerate() {
                    for v in 0..256 {
                        if v >> (j % 8) & 1 == 1 {
                            lut[j / 8][v] |= 1 << m;
                        }
                    }
                }
                lut
            })
            .collect()
    }

    /// The conjugation lookups and the stored moves of the searched moves, of the reduced
    /// table (27-move masks; the fourth byte's lookups read only its three mask bits).
    fn conj27_tables() -> (Vec<[[u32; 256]; 4]>, Vec<[u8; 32]>) {
        Self::conj_moves()
            .iter()
            .map(|pi| {
                let mut lut = [[0u32; 256]; 4];
                let mut inv = [0u8; 32];
                for (m, &j) in pi.iter().enumerate() {
                    let j = fold(j);
                    inv[m] = j as u8;
                    for v in 0..256 {
                        if v >> (j % 8) & 1 == 1 {
                            lut[j / 8][v] |= 1 << m;
                        }
                    }
                }
                (lut, inv)
            })
            .unzip()
    }

    /// A reduced table with these entries (and its lookups).
    fn reduced(red: HugeBuf<[u64; 2]>) -> MaskTable {
        let (conj27, inv27) = Self::conj27_tables();
        MaskTable {
            red,
            conj27,
            inv27,
            ..Self::empty(MaskKind::Reduced)
        }
    }

    /// A 27-bit stored mask (the low bits of `x`) conjugated to the searched state's moves.
    #[inline(always)]
    fn conj27(c: &[[u32; 256]; 4], x: u64) -> u32 {
        c[0][(x & 255) as usize]
            | c[1][(x >> 8 & 255) as usize]
            | c[2][(x >> 16 & 255) as usize]
            | c[3][(x >> 24 & 255) as usize]
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

    /// Prefetch the reduced table's entry `j`.
    #[inline(always)]
    pub fn prefetch_entry(&self, j: u64) {
        crate::table::prefetch(unsafe { self.red.as_ptr().add(j as usize) });
    }

    /// Whether a node at staging distance = moves left (>= 1) with reduced index `i` (as
    /// above) and odd (`odd`) or even wing parity can still reach an endpoint: its entry's
    /// viability bit (see [`MaskTable::add_viable`]; reduced table only).
    #[inline(always)]
    pub fn viable(&self, i: u64, odd: bool) -> bool {
        self.red[i as usize][0] >> (VIABLE_SHIFT + odd as u32) & 1 == 1
    }

    /// The child bits of reduced entry `i` for a node with odd (`odd`) or even wing parity
    /// (bit j: the child via the move whose stored move is j, see [`MaskTable::stored_moves`])
    /// and its live bits (bits 0..18, the outer moves); see [`MaskTable::add_live`] and
    /// [`MaskTable::child_live`].
    #[inline(always)]
    pub fn child_bits(&self, i: u64, odd: bool) -> (u64, u64) {
        let [w0, w1] = self.red[i as usize];
        (
            w1 >> (ODD_SHIFT * odd as u32),
            w0 >> LIVE0_SHIFT | w1 >> LIVE1_SHIFT << 8,
        )
    }

    /// Per searched move, its stored move, for a node with symmetry `sym` (reduced table).
    #[inline(always)]
    pub fn stored_moves(&self, sym: u8) -> &[u8; 32] {
        unsafe { self.inv27.get_unchecked(sym as usize) }
    }

    /// Can the child via move `m` of a node (its [`MaskTable::child_bits`] `bits` and
    /// [`MaskTable::stored_moves`] `stored`) still reach an endpoint, if the child is at
    /// distance = moves left (>= 1)? Not if its wing parity is not viable, nor if every move
    /// that decreases its distance is one the canonical order excludes after `m`: a move of
    /// `m`'s axis on an earlier face or on `m`'s (the child via a wide move: none on another
    /// axis; via an outer move: none on another axis and none in `pre` on `m`'s axis on a later
    /// face). `pre`: moves y such that every child's decreasing move y on `m`'s axis is in it
    /// (y commutes with `m`, so the node after y and `m` is the node after `m` and y, one move
    /// further from staged centres than the node after y can be: for a child one closer than
    /// the node, y must decrease the node's distance; one as far, keep or decrease it).
    #[inline(always)]
    pub fn child_live(bits: (u64, u64), stored: &[u8; 32], m: usize, pre: u32) -> bool {
        debug_assert!(m < N_MOVES);
        let j = unsafe { *stored.get_unchecked(m) } as u32;
        bits.0 >> j & 1 == 1
            && (m >= N_OUTER
                || bits.1 >> j & 1 == 1
                || pre & unsafe { *LATER_SAME_AXIS.get_unchecked(m) } != 0)
    }

    /// [`MaskTable::child_live`] for every move at once, without a branch per move: the moves
    /// whose children (if at distance = moves left) can still reach an endpoint, for a node
    /// with symmetry `sym` and child bits `bits`. The stored bits are mapped to the searched
    /// moves by the mask conjugation, and `pre`'s moves on a later face of each outer move's
    /// axis are found per face (face f's later faces on its axis are f + 3 and f + 6).
    #[inline(always)]
    pub fn live_moves(&self, bits: (u64, u64), sym: u8, pre: u32) -> u32 {
        let c = unsafe { self.conj27.get_unchecked(sym as usize) };
        let viable = Self::conj27(c, bits.0);
        let live = Self::conj27(c, bits.1);
        // Per face (bit 3f), whether `pre` has a move on it; then per outer face whether a
        // later face of its axis has one, spread to the face's three moves.
        let t = pre | pre >> 1 | pre >> 2;
        let q = t & 0x124_9249;
        let later = ((q >> 9 | q >> 18) * 7) & ((1 << N_OUTER) - 1);
        viable & (!((1u32 << N_OUTER) - 1) | live | later)
    }

    /// The 27-move masks of the moves that decrease and that keep the staging distance, of
    /// the state with reduced index `i` and symmetry `sym` (see [`MaskTable::resolve`];
    /// reduced table only).
    #[inline(always)]
    pub fn get_dec_eq(&self, i: u64, sym: u8) -> (u32, u32) {
        let w = self.red[i as usize][0];
        let c = unsafe { self.conj27.get_unchecked(sym as usize) };
        (Self::conj27(c, w), Self::conj27(c, w >> EQ_SHIFT))
    }

    /// The 27-move mask of state `s` (index `i` and symmetry `sym` as above); the virtual
    /// table also serves the per-pair searches (`pair`: the pattern table's distance).
    #[inline(always)]
    pub fn get(&self, t: &P1Tables, i: u64, sym: u8, s: &P1State, pair: usize) -> u32 {
        match self.kind {
            MaskKind::Big => self.big[i as usize],
            MaskKind::Reduced => {
                let c = unsafe { self.conj27.get_unchecked(sym as usize) };
                Self::conj27(c, self.red[i as usize][0])
            }
            MaskKind::Pair => {
                let e = if self.lazy {
                    let pt = t.pat.as_ref().unwrap();
                    let g = pt.red_state(&t.staging, i as usize);
                    pair_mask_of(t, &g, pt.table.get(i), &moves36())
                } else {
                    // SAFETY: 5 i + 8 is within the buffer (3 bytes of padding).
                    unsafe {
                        (self.pair.as_ptr().add(5 * i as usize) as *const u64).read_unaligned()
                    }
                };
                // (The conjugation lookups of the fifth byte read only its four mask bits.)
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

    /// Set the viability bits of the reduced table's entries (bits 54 and 55 of word 0, for
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
    /// representative's are the searched state's. Dw, Lw and Bw (quarter turns: wide quarter
    /// turns) lead to the same paths as Uw, Rw and Fw up to a rotation (see [`fold`]), so the
    /// paths over the 27 stored moves decide. Generated from the masks alone by memoised
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
            let mut dec = c.red[i as usize][0] & MASK27;
            let mut v = false;
            while dec != 0 && !v {
                let m = dec.trailing_zeros() as usize;
                dec &= dec - 1;
                let quarter = crate::moves::WIDE_QUARTER >> m & 1 == 1;
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
            if *e & MASK27 != 0 {
                dead[0] += (bits & 1 == 0) as u64;
                dead[1] += (bits & 2 == 0) as u64;
            }
            *e = *e & !(3 << VIABLE_SHIFT) | bits << VIABLE_SHIFT;
        }
        dead
    }

    /// Set the child and live bits of the reduced table's entries (after the viability bits,
    /// see [`MaskTable::add_viable`]), and return how many (entry, move) pairs have their child
    /// bit clear for even and odd wing parity, and their live bit clear (outer moves).
    ///
    /// For the child via each stored move j of the representative (its 27 moves): its live
    /// bit says whether the child has a distance-decreasing move on another axis than j's
    /// (which the canonical order always allows after j; outer moves, bits 0..18); its child
    /// bit for a wing parity p of the node, whether its viability bit for its own parity (p,
    /// flipped by a wide quarter turn) is set and, for a wide move, its live bit too (the
    /// canonical order allows no move of a wide move's axis after it). So a child at distance
    /// = moves left (>= 1) whose child bit is clear has no endpoint below it, and neither
    /// has one whose decreasing moves are all on j's axis on earlier faces (see
    /// [`MaskTable::child_live`]). The 48 symmetries map children to children and axes to
    /// axes, so the bits of a searched state's move are those of its stored move.
    pub fn add_live(&mut self, st: &Staging) -> [u64; 3] {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        let n = N_CLASSES_A * N_SPLIT;
        let mv = moves36();
        // Each entry's own bits are written by one thread only, while others read the bits
        // computed before (masks and viability; disjoint bits of word 0): atomics.
        let words = self.red.as_mut_slice().as_mut_ptr() as usize;
        let word = |k: usize| unsafe { &*(words as *const AtomicU64).add(k) };
        let conj27 = &self.conj27;
        let clear: [std::sync::atomic::AtomicU64; 3] = Default::default();
        crate::par::for_chunks(n, 1 << 12, |r| {
            let mut cl = [0u64; 3];
            for i in r {
                let (ud, rl) = st.rep_of(i as u64);
                let flags = flags_of(ud, rl);
                let (mut child, mut live) = ([0u64; 2], 0u64);
                for (j, g) in mv[..N_MOVES].iter().enumerate() {
                    let g = flags.gather(g);
                    let info = unsafe { *st.info(flag_masks(&g).0) };
                    let w = word(2 * st.fast_index_flags(info, &g) as usize).load(Relaxed);
                    // The child's decreasing moves, as moves of this state's frame.
                    let dec = Self::conj27(&conj27[(info & 255) as usize], w);
                    let other = dec & OTHER_AXES[j] != 0;
                    let quarter = crate::moves::WIDE_QUARTER >> j & 1;
                    for (p, c) in child.iter_mut().enumerate() {
                        let viable = w >> (VIABLE_SHIFT + (p as u32 ^ quarter)) & 1 == 1;
                        if viable && (j < N_OUTER || other) {
                            *c |= 1 << j;
                        }
                    }
                    if j < N_OUTER && other {
                        live |= 1 << j;
                    }
                }
                cl[0] += N_MOVES as u64 - child[0].count_ones() as u64;
                cl[1] += N_MOVES as u64 - child[1].count_ones() as u64;
                cl[2] += N_OUTER as u64 - live.count_ones() as u64;
                word(2 * i + 1).store(
                    child[0] | child[1] << ODD_SHIFT | (live >> 8) << LIVE1_SHIFT,
                    Relaxed,
                );
                word(2 * i).fetch_or((live & 255) << LIVE0_SHIFT, Relaxed);
            }
            for (c, x) in clear.iter().zip(cl) {
                c.fetch_add(x, Relaxed);
            }
        });
        clear.map(|c| c.into_inner())
    }

    /// The reduced table (masks only), from the staging table (either).
    pub fn generate_reduced(t: &P1Tables) -> MaskTable {
        let n = N_CLASSES_A * N_SPLIT;
        let mut red = HugeBuf::<[u64; 2]>::zeroed(n);
        let ptr = red.as_mut_slice().as_mut_ptr() as usize;
        let mv = moves36();
        crate::par::for_chunks(n, 1 << 12, |r| {
            for i in r {
                let (ud, rl) = t.staging.rep_of(i as u64);
                let (dec, eq) = masks_of(t, ud, rl, &mv[..N_MOVES]);
                // SAFETY: each index is written by one thread only, within the buffer.
                unsafe { *(ptr as *mut [u64; 2]).add(i) = [dec | eq << EQ_SHIFT, 0] };
            }
        });
        Self::reduced(red)
    }
}

/// Load (or generate from the staging table, and save) a mask table in `dir`: the large one
/// (`big`, 18.9 GB, `p1-mask-big.tab`; from the large staging table) or the reduced one
/// (1.6 GB, `p1-mask-live.tab`; from either staging table). `EGGG_P1_MASK_VIRTUAL` set: no table,
/// masks computed when needed (slow; tests).
pub fn load(dir: &std::path::Path, t: &P1Tables, big: bool, verbose: bool) -> MaskTable {
    if std::env::var_os("EGGG_P1_MASK_VIRTUAL").is_some() {
        return MaskTable::virtual_table();
    }
    if t.pat.is_some() {
        return load_pair(dir, t, verbose);
    }
    // (The reduced tables of earlier versions: "p1-mask", 36-move masks without the equal
    // masks; "p1-mask2", 36-move masks and viability bits, no child or live bits.)
    let name = if big { "p1-mask-big" } else { "p1-mask-live" };
    let path = dir.join(format!("{name}.tab"));
    let mut m = MaskTable::empty(if big {
        MaskKind::Big
    } else {
        MaskKind::Reduced
    });
    let bytes = if big {
        m.big = HugeBuf::zeroed(Staging::big_len());
        m.big.as_bytes_mut()
    } else {
        m.red = HugeBuf::zeroed(N_CLASSES_A * N_SPLIT);
        m.red.as_bytes_mut()
    };
    if crate::table::load(&path, name, bytes).is_ok() {
        if !big {
            return MaskTable::reduced(m.red);
        }
        return m;
    }
    drop(m);
    if verbose {
        eprintln!(
            "generating the phase-1 move-mask table ({})...",
            if big { "18.9 GB" } else { "1.6 GB" }
        );
    }
    let mut m = if big {
        MaskTable::generate_big(t)
    } else {
        MaskTable::generate_reduced(t)
    };
    if !big {
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
        let t0 = std::time::Instant::now();
        let clear = m.add_live(&t.staging);
        if verbose {
            let n = m.red.len() as f64;
            eprintln!(
                "phase-1 mask child bits in {:.1} s (clear: {:.1}% / {:.1}% for even / odd \
                 wing parity, live bits {:.1}%)",
                t0.elapsed().as_secs_f64(),
                100.0 * clear[0] as f64 / (n * N_MOVES as f64),
                100.0 * clear[1] as f64 / (n * N_MOVES as f64),
                100.0 * clear[2] as f64 / (n * N_OUTER as f64)
            );
        }
    }
    let bytes = if big {
        m.big.as_bytes()
    } else {
        m.red.as_bytes()
    };
    crate::table::save(&path, name, bytes).expect("saving table");
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

    /// [`MaskTable::live_moves`] is [`MaskTable::child_live`] for every move, for every
    /// symmetry, random child and live bits and random (and all) `pre` masks.
    #[test]
    fn live_moves_match_child_live() {
        let (conj27, inv27) = MaskTable::conj27_tables();
        let mut t = MaskTable::empty(MaskKind::Reduced);
        t.conj27 = conj27;
        t.inv27 = inv27;
        let mut rng = rand::thread_rng();
        for sym in 0..t.inv27.len() {
            for i in 0..2000 {
                let bits: (u64, u64) = (
                    rand::Rng::r#gen::<u64>(&mut rng),
                    rand::Rng::r#gen::<u64>(&mut rng) & ((1 << N_OUTER) - 1),
                );
                let pre = if i % 4 == 0 {
                    u32::MAX
                } else {
                    rand::Rng::r#gen::<u32>(&mut rng)
                        & MASK27 as u32
                        & rand::Rng::r#gen::<u32>(&mut rng)
                };
                let want = (0..N_MOVES).fold(0u32, |a, m| {
                    a | (MaskTable::child_live(bits, t.stored_moves(sym as u8), m, pre) as u32) << m
                });
                assert_eq!(t.live_moves(bits, sym as u8, pre), want, "sym {sym}");
            }
        }
    }

    /// The large and reduced tables' masks (as computed by their generators, from either
    /// staging table, for random states) are the moves that decrease and keep the distance;
    /// the representative's masks over the 36 moves have the same bits for Dw, Lw and Bw as for
    /// Uw, Rw and Fw ([`fold`]) (needs the large staging table in ./tables; skipped without it).
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
        let (conj, _) = MaskTable::conj27_tables();
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
                // The reduced table's entry: the 27-move masks of the symmetric image (from
                // either staging table).
                let info = unsafe { *t.staging.info(ud) };
                let j = t.staging.fast_index_flags(info, &s.centre_flags());
                let (ru, rr) = t.staging.rep_of(j);
                let e = masks_of(&t, ru, rr, &mv[..N_MOVES]);
                assert_eq!(masks_of(&small, ru, rr, &mv[..N_MOVES]), e);
                let c = &conj[(info & 255) as usize];
                let f = |x: u64| MaskTable::conj27(c, x);
                assert_eq!((f(e.0), f(e.1)), (want, want_eq));
                let e36 = masks_of(&t, ru, rr, &mv);
                for x in [e36.0, e36.1] {
                    assert_eq!(x >> N_MOVES, x >> 18 & 511, "Dw Lw Bw fold onto Uw Rw Fw");
                }
                assert_eq!((e36.0 & MASK27, e36.1 & MASK27), e);
                let m = (k * 5 + 1) % N_MOVES;
                s = s.apply(m, &t);
                d = t.distance(&s);
            }
        }
    }

    /// The reduced table's entries, as the search reads them (by the searched state's index
    /// and symmetry), give the moves that decrease and keep the staging distance, by the
    /// symmetry-reduced staging table (needs `p1-mask-live.tab` in ./tables, made by
    /// `eggg-solver gen --tier large`; skipped without it).
    #[test]
    fn reduced_entries_give_child_distances() {
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-mask-live.tab").exists() {
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

    /// The child and live bits, as the search reads them (by the searched state's reduced
    /// index and symmetry, through its stored moves), agree with a brute force over the
    /// symmetry-reduced staging table, for every move and both wing parities, from random
    /// states and from states a few moves from staged centres (many of them with symmetries);
    /// and [`MaskTable::child_live`] never drops a child with an allowed decreasing move and a
    /// viable wing parity, and is exact after a wide move (needs `p1-mask-live.tab` in ./tables;
    /// skipped without it).
    #[test]
    fn child_and_live_bits_match_brute_force() {
        use crate::moves::{ALLOWED_AFTER, WIDE_QUARTER};
        use crate::table::Mod3Table;
        use std::collections::HashMap;
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-mask-live.tab").exists() {
            return;
        }
        let t = crate::solver::load_p1(dir, false, 8, false);
        let mt = load(dir, &t, false, false);
        type Memo = HashMap<(u64, bool), bool>;
        // A path of d distance-decreasing moves to staged centres with even wing parity, its
        // last move a wide quarter turn, from `s` at distance d with wing parity `odd`?
        fn viable(t: &P1Tables, s: &P1State, d: u8, odd: bool, memo: &mut Memo) -> bool {
            if d == 0 {
                return false;
            }
            let (ud, rl) = s.masks();
            let key = (t.staging.index(ud, rl), odd);
            if let Some(&v) = memo.get(&key) {
                return v;
            }
            let mut v = false;
            for m in 0..N_MOVES {
                let c = s.apply(m, t);
                let dc = Mod3Table::decode(d, t.value(&c));
                if dc + 1 != d {
                    continue;
                }
                let quarter = WIDE_QUARTER >> m & 1 == 1;
                v = if dc == 0 {
                    quarter && odd
                } else {
                    viable(t, &c, dc, odd ^ quarter, memo)
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
        let (mut symmetric, mut dropped, mut kept) = (0, 0, 0);
        for k in 0..600 {
            let mut s = P1State::from_cube(&Cube::SOLVED);
            if k % 3 == 0 {
                s = P1State::from_cube(&Cube::random(&mut rng));
            } else {
                for _ in 0..1 + k % 8 {
                    s = s.apply(rand::Rng::gen_range(&mut rng, 0..N_MOVES), &t);
                }
            }
            let d = t.distance(&s);
            let i = mt.pre_index(&t, &s, crate::phase1::ALL_PAIRS);
            let (j, sym) = mt.resolve(&t, i, &s, crate::phase1::ALL_PAIRS);
            let info = unsafe { *t.staging.info(s.masks().0) };
            symmetric += !t.staging.stab[(info >> 8) as usize].is_empty() as usize;
            let stored = mt.stored_moves(sym);
            let (dec, eq) = (0..N_MOVES).fold((0u32, 0u32), |(a, b), m| {
                let dc = Mod3Table::decode(d, t.value(&s.apply(m, &t)));
                (a | ((dc + 1 == d) as u32) << m, b | ((dc == d) as u32) << m)
            });
            for odd in [false, true] {
                let bits = mt.child_bits(j, odd);
                for m in 0..N_MOVES {
                    let c = s.apply(m, &t);
                    let dc = Mod3Table::decode(d, t.value(&c));
                    let cdec = (0..N_MOVES).fold(0u32, |a, y| {
                        let dy = Mod3Table::decode(dc, t.value(&c.apply(y, &t)));
                        a | ((dy + 1 == dc) as u32) << y
                    });
                    let codd = odd ^ (WIDE_QUARTER >> m & 1 == 1);
                    let v = viable(&t, &c, dc, codd, &mut memo);
                    let other = cdec & OTHER_AXES[m] != 0;
                    let jm = stored[m] as u32;
                    assert_eq!(
                        bits.0 >> jm & 1 == 1,
                        v && (m < N_OUTER || other),
                        "child bit"
                    );
                    if m < N_OUTER {
                        assert_eq!(bits.1 >> jm & 1 == 1, other, "live bit");
                    }
                    // The search's test, with the node's moves that can precede a child's
                    // decreasing move on m's axis (by the child's distance).
                    let pre = if dc + 1 == d {
                        dec
                    } else if dc == d {
                        dec | eq
                    } else {
                        u32::MAX
                    };
                    let live = MaskTable::child_live(bits, stored, m, pre);
                    let exact = v && cdec & ALLOWED_AFTER[m] != 0;
                    assert!(live || !exact, "a live child dropped");
                    if m >= N_OUTER {
                        assert_eq!(live, exact);
                    }
                    if dc >= 1 {
                        dropped += !live as usize;
                        kept += live as usize;
                    }
                }
            }
        }
        assert!(symmetric > 20, "{symmetric} states with symmetries");
        assert!(dropped > 1000 && kept > 1000, "{dropped} / {kept}");
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
