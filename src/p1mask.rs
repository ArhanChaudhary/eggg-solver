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
}

pub struct MaskTable {
    pub kind: MaskKind,
    /// Batches also read the masks of nodes one move from the end, so that only their
    /// staging last moves get the wing tests (`EGGG_P1_BATCH_LEAVES` set).
    pub batch_leaves: bool,
    big: HugeBuf<u32>,
    red: HugeBuf<u64>,
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
            big: HugeBuf::zeroed(1),
            red: HugeBuf::zeroed(1),
            conj: vec![],
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
            MaskKind::Virtual => {}
        }
    }

    /// Prefetch what the mask of the node with large staging index `i` needs first.
    #[inline(always)]
    pub fn prefetch(&self, t: &P1Tables, i: u64) {
        match self.kind {
            MaskKind::Big => {
                crate::table::prefetch(unsafe { self.big.as_ptr().add(i as usize) });
            }
            MaskKind::Reduced => {
                crate::table::prefetch(t.staging.info_of_rank(i / N_SPLIT as u64));
            }
            MaskKind::Virtual => {}
        }
    }

    /// The reduced table's index and symmetry for the node with large staging index `i`
    /// (its class info prefetched), the entry prefetched.
    #[inline(always)]
    pub fn resolve(&self, t: &P1Tables, i: u64, s: &P1State) -> (u64, u8) {
        let st = &t.staging;
        let info = unsafe { *st.info_of_rank(i / N_SPLIT as u64) };
        let j = st.fast_index_flags(info, &s.centre_flags());
        crate::table::prefetch(unsafe { self.red.as_ptr().add(j as usize) });
        (j, info as u8)
    }

    /// The 27-move mask of state `s` (index `i` and symmetry `sym` as above).
    #[inline(always)]
    pub fn get(&self, t: &P1Tables, i: u64, sym: u8, s: &P1State) -> u32 {
        match self.kind {
            MaskKind::Big => self.big[i as usize],
            MaskKind::Reduced => {
                let e = self.red[i as usize];
                let c = unsafe { self.conj.get_unchecked(sym as usize) };
                c[0][(e & 255) as usize]
                    | c[1][(e >> 8 & 255) as usize]
                    | c[2][(e >> 16 & 255) as usize]
                    | c[3][(e >> 24 & 255) as usize]
                    | c[4][(e >> 32 & 255) as usize]
            }
            MaskKind::Virtual => {
                let (ud, rl) = s.masks();
                mask_of(t, ud, rl, &moves36()[..N_MOVES]) as u32
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
                let want = v.get(&t, 0, 0, &s);
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
}
