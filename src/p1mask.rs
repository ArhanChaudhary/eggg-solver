//! Phase-1 move masks: per staging state, the moves that decrease its staging distance.
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
//! outer block moves (8 bytes, 0.8 GB): the searched state's 27-move mask is the stored mask
//! with each move conjugated by that symmetry (five byte lookups).
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
/// The 36 outer block moves: the solver's 27, then Dw, Lw, Bw (each power).
const N36: usize = 36;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MaskKind {
    /// 27-bit masks by the large staging index (18.9 GB).
    Big,
    /// 36-bit masks by the reduced staging index (0.8 GB).
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
    big: HugeBuf<u32>,
    red: HugeBuf<u64>,
    /// The pair table: 5 bytes per entry (and 3 bytes of padding).
    pair: HugeBuf<u8>,
    /// The pair table's entries are computed when read, from the pattern table (tests).
    lazy: bool,
    /// Per symmetry, per byte of a stored 36-bit mask: the searched state's 27-bit mask.
    conj: Vec<[[u32; 256]; 5]>,
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

/// The mask over `moves` of the state with U/D mask `ud` and R/L mask `rl`, from the large
/// staging table.
fn mask_of(t: &P1Tables, ud: u32, rl: u32, moves: &[V32]) -> u64 {
    let st = &t.staging;
    let v = t.table.get(st.big_index(ud, rl));
    if v == 3 {
        return 0;
    }
    let flags = flags_of(ud, rl);
    let mut idx = [0u64; N36];
    for (e, mv) in idx.iter_mut().zip(moves) {
        let (a, b, c) = flag_masks(&flags.gather(mv));
        *e = st.big_index_n(a, b, c);
        t.table.prefetch(*e);
    }
    let want = (v + 2) % 3;
    let mut mask = 0;
    for (m, &e) in idx[..moves.len()].iter().enumerate() {
        if t.table.get(e) == want {
            mask |= 1 << m;
        }
    }
    mask
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
            big: HugeBuf::zeroed(1),
            red: HugeBuf::zeroed(1),
            pair: HugeBuf::zeroed(1),
            lazy: false,
            conj: vec![],
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

    /// The conjugation lookups of the reduced table.
    fn conj_tables() -> Vec<[[u32; 256]; 5]> {
        let mv = moves36();
        let syms = centre_syms();
        let mut out = vec![];
        for p in &syms {
            // pi(m): the move whose centre permutation is p o C_m o p^-1, i.e. the move m
            // seen on the symmetric image of the state.
            let mut lut = [[0u32; 256]; 5];
            for (m, cm) in mv.iter().enumerate().take(N_MOVES) {
                let j = (0..N36)
                    .find(|&j| (0..24).all(|i| mv[j].0[p[i] as usize] == p[cm.0[i] as usize]))
                    .expect("conjugate move");
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
        crate::table::prefetch(unsafe { self.red.as_ptr().add(j as usize) });
        (j, info as u8)
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
                    self.red[i as usize]
                };
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
                    let m = mask_of(t, ud, rl, &mv[..N_MOVES]) as u32;
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

    /// The reduced table, from the large staging table (`t.big`).
    pub fn generate_reduced(t: &P1Tables) -> MaskTable {
        assert!(t.big);
        let n = N_CLASSES_A * N_SPLIT;
        let mut red = HugeBuf::<u64>::zeroed(n);
        let ptr = red.as_mut_slice().as_mut_ptr() as usize;
        let mv = moves36();
        crate::par::for_chunks(n, 1 << 12, |r| {
            for i in r {
                let (ud, rl) = t.staging.rep_of(i as u64);
                // SAFETY: each index is written by one thread only, within the buffer.
                unsafe { *(ptr as *mut u64).add(i) = mask_of(t, ud, rl, &mv) };
            }
        });
        MaskTable {
            red,
            conj: Self::conj_tables(),
            ..Self::empty(MaskKind::Reduced)
        }
    }
}

/// Load (or generate from the large staging table, and save) a mask table in `dir`: the
/// large one (`big`, 18.9 GB) or the reduced one (0.8 GB). `EGGG_P1_MASK_VIRTUAL` set: no
/// table, masks computed when needed (slow; tests).
pub fn load(dir: &std::path::Path, t: &P1Tables, big: bool, verbose: bool) -> MaskTable {
    if std::env::var_os("EGGG_P1_MASK_VIRTUAL").is_some() {
        return MaskTable::virtual_table();
    }
    if t.pat.is_some() {
        return load_pair(dir, t, verbose);
    }
    let name = if big { "p1-mask-big" } else { "p1-mask" };
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
            m.conj = MaskTable::conj_tables();
        }
        return m;
    }
    drop(m);
    if verbose {
        eprintln!(
            "generating the phase-1 move-mask table ({})...",
            if big { "18.9 GB" } else { "0.8 GB" }
        );
    }
    let m = if big {
        MaskTable::generate_big(t)
    } else {
        MaskTable::generate_reduced(t)
    };
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

    /// The large and reduced tables' masks (as computed by their generators, for random
    /// states) are the distance-decreasing moves (needs the large staging table in ./tables;
    /// skipped without it).
    #[test]
    fn masks_are_decreasing_moves() {
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-staging-big.tab").exists() {
            return;
        }
        let t = crate::solver::load_p1(dir, true, 8, false);
        let v = MaskTable::virtual_table();
        let mv = moves36();
        let conj = MaskTable::conj_tables();
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let mut s = P1State::from_cube(&Cube::random(&mut rng));
            for k in 0..12 {
                let want = v.get(&t, 0, 0, &s, crate::phase1::ALL_PAIRS);
                let (ud, rl) = s.masks();
                assert_eq!(mask_of(&t, ud, rl, &mv[..N_MOVES]) as u32, want);
                // The reduced table's entry: the 36-move mask of the symmetric image.
                let info = unsafe { *t.staging.info(ud) };
                let j = t.staging.fast_index_flags(info, &s.centre_flags());
                let (ru, rr) = t.staging.rep_of(j);
                let e = mask_of(&t, ru, rr, &mv);
                let c = &conj[(info & 255) as usize];
                let got = (0..5).fold(0, |a, k| a | c[k][(e >> (8 * k) & 255) as usize]);
                assert_eq!(got, want);
                s = s.apply((k * 5 + 1) % N_MOVES, &t);
            }
        }
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
