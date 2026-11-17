//! Phase-1 pattern tables: for one colour pair, the distance to staged centres with that pair
//! on some axis in a pattern phase 2 can solve in that axis's frame.
//!
//! The phase-1 goal for an endpoint whose special axis holds colour pair c is reached exactly
//! when this distance (for pair c) is 0, so a search per pair pruned by it visits only nodes
//! that can still reach such an endpoint: deep in the phase-1 tree almost every node is at
//! exact staging distance = moves left, and only ~1 in 5 of those (per pair) can also fix the
//! pattern without extra moves. The union of the three pair searches' endpoints (each with
//! the frames whose special axis holds its pair) is the plain search's.
//!
//! The coordinate: the 8 positions of the pair (8 of 24), which 4 of them hold its first
//! colour (unlabelled: 35), and the split of the other 16 positions between the other two
//! pairs (unlabelled: 6435). The goal is invariant under the 48 symmetries (the allowed
//! patterns are the same in every frame up to symmetry, see `patterns_are_symmetric`), and,
//! being rotation-invariant, has the same distance with the solver's 27 moves as with all 36;
//! the generated table is reduced by the 48 symmetries like the staging table (15582 classes
//! of the pair's positions x 35 x 6435 = 3.5e9 entries, 0.88 GB at two bits); the large
//! variant has no reduction (735471 x 35 x 6435 = 1.66e11 entries, 41.4 GB) and is filled
//! from the reduced one.

use crate::comb::comb_unrank;
use crate::moves::{MOVE_PERMS, N_MOVES};
use crate::phase1::{Frame, P1State, Staging, pdep, pext};
use crate::simd::V32;
use crate::table::{HugeBuf, Mod3Table};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const MASK24: u32 = 0xff_ffff;
/// Unlabelled splits of the pair's 8 positions between its two colours.
pub const N_X: usize = 35;
/// Unlabelled splits of the other 16 positions between the other two pairs.
pub const N_Q: usize = crate::phase1::N_SPLIT;
const N_XQ: usize = N_X * N_Q;
pub const N_RED: usize = crate::phase1::N_CLASSES_A * N_XQ;
pub const N_BIG: usize = 735_471 * N_XQ;

pub struct PatTable {
    pub table: Mod3Table,
    /// The table is the large one (no symmetry reduction).
    pub big: bool,
    /// Unlabelled rank of a 4-of-8 mask (x and its complement share a rank).
    unl8: [u8; 256],
    unl8_inv: [u8; N_X],
    /// Per pair: centre code -> flags (bit 7: the pair, bit 6: its first colour, bit 5: the
    /// next pair).
    luts: [V32; 3],
}

const LOW24: V32 = {
    let mut t = [0u8; 32];
    let mut i = 0;
    while i < 24 {
        t[i] = 0xff;
        i += 1;
    }
    V32(t)
};

/// The pair, first-colour and next-pair position masks from flags.
#[inline(always)]
pub fn flag_masks(g: &V32) -> (u32, u32, u32) {
    let p = crate::simd::movemask(g);
    let x = crate::simd::movemask(&crate::simd::add(g, g));
    let q = crate::simd::movemask(&crate::simd::shl::<2>(g));
    (p, x, q)
}

/// Flags (see [`PatTable::flags`]) from the three masks.
fn masks_flags(p: u32, x: u32, q: u32) -> V32 {
    let mut v = V32::ZERO;
    for i in 0..24 {
        v.0[i] = ((p >> i & 1) << 7 | (x >> i & 1) << 6 | (q >> i & 1) << 5) as u8;
    }
    v
}

impl PatTable {
    pub fn new(table: Mod3Table, big: bool) -> PatTable {
        let mut unl8 = [u8::MAX; 256];
        let mut unl8_inv = [0u8; N_X];
        let mut k = 0;
        for x in 0u32..256 {
            if x.count_ones() == 4 && unl8[x as usize] == u8::MAX {
                unl8[x as usize] = k as u8;
                unl8[(!x & 255) as usize] = k as u8;
                unl8_inv[k] = x as u8;
                k += 1;
            }
        }
        assert_eq!(k, N_X);
        let luts = std::array::from_fn(|c| {
            let mut t = [0u8; 32];
            for (i, e) in t.iter_mut().enumerate() {
                let code = i & 15;
                if code >= 6 {
                    continue;
                }
                if code >> 1 == c {
                    *e |= 0x80;
                    if code & 1 == 0 {
                        *e |= 0x40;
                    }
                }
                if code >> 1 == (c + 1) % 3 {
                    *e |= 0x20;
                }
            }
            V32(t)
        });
        PatTable {
            table,
            big,
            unl8,
            unl8_inv,
            luts,
        }
    }

    pub fn len(big: bool) -> usize {
        if big { N_BIG } else { N_RED }
    }

    /// Per centre position, bit 7 for colour pair `c`, bit 6 for its first colour, bit 5 for
    /// the next pair (zero at positions 24..).
    #[inline(always)]
    pub fn flags(&self, centres: &V32, c: usize) -> V32 {
        crate::simd::and(&crate::simd::shuf(&self.luts[c], centres), &LOW24)
    }

    /// Index into the large table.
    #[inline(always)]
    pub fn big_index(&self, st: &Staging, g: &V32) -> u64 {
        let (p, x, q) = flag_masks(g);
        (st.rank24(p) as usize * N_XQ
            + self.unl8[pext(x, p) as u8 as usize] as usize * N_Q
            + st.unl[pext(q, !p & MASK24) as u16 as usize] as usize) as u64
    }

    /// Index into the reduced table, given the pair's class info (one stabiliser variant;
    /// the table holds the same value in all of them).
    #[inline(always)]
    pub fn red_index(&self, st: &Staging, info: u32, g: &V32) -> u64 {
        let class = (info >> 8) as usize;
        let h = g.gather(&st.sym_gather[(info & 255) as usize]);
        let (p, x, q) = flag_masks(&h);
        (class * N_XQ
            + self.unl8[pext(x, p) as u8 as usize] as usize * N_Q
            + st.unl[pext(q, !p & MASK24) as u16 as usize] as usize) as u64
    }

    /// The canonical reduced index (the minimum over the class's stabiliser).
    fn canonical(&self, st: &Staging, g: &V32) -> u64 {
        let p = crate::simd::movemask(g);
        let info = st.a_info[st.rank24(p) as usize];
        let class = (info >> 8) as usize;
        let h = g.gather(&st.sym_gather[(info & 255) as usize]);
        let rest = |h: &V32| {
            let (p, x, q) = flag_masks(h);
            self.unl8[pext(x, p) as u8 as usize] as usize * N_Q
                + st.unl[pext(q, !p & MASK24) as u16 as usize] as usize
        };
        let mut r = rest(&h);
        for &t in &st.stab[class] {
            r = r.min(rest(&h.gather(&st.sym_gather[t as usize])));
        }
        (class * N_XQ + r) as u64
    }

    /// Index of a state for pair `c` (either table).
    #[inline(always)]
    pub fn index(&self, st: &Staging, centres: &V32, c: usize) -> u64 {
        let g = self.flags(centres, c);
        if self.big {
            self.big_index(st, &g)
        } else {
            let info = st.a_info[st.rank24(crate::simd::movemask(&g)) as usize];
            self.red_index(st, info, &g)
        }
    }

    #[inline(always)]
    pub fn value(&self, st: &Staging, s: &P1State, c: usize) -> u8 {
        self.table.get(self.index(st, &s.centres, c))
    }

    /// Flags of the reduced table's state `i` (its class representative).
    fn red_state(&self, st: &Staging, i: usize) -> V32 {
        let class = i / N_XQ;
        let x = i / N_Q % N_X;
        let q = i % N_Q;
        let p = st.class_rep[class];
        masks_flags(
            p,
            pdep(self.unl8_inv[x] as u32, p),
            pdep(st.unl_inv[q] as u32, !p & MASK24),
        )
    }

    /// The goal states' flags: each frame's special axis holding the pair in an allowed
    /// pattern, the other pairs staged.
    fn goal_flags() -> Vec<V32> {
        let ax = Staging::axis_masks();
        let mut out = vec![];
        for f in 0..3 {
            let fr = Frame::new(f);
            let p: u32 = fr.rl_pos.iter().map(|&q| 1u32 << q).sum();
            let other = *ax.iter().find(|&&a| a != p).unwrap();
            for pat in 0u32..256 {
                if fr.rl_ok[pat as usize >> 6] >> (pat & 63) & 1 == 0 {
                    continue;
                }
                let mut x = 0;
                for i in 0..8 {
                    if pat >> i & 1 == 1 {
                        x |= 1 << fr.rl_pos[i];
                    }
                }
                out.push(masks_flags(p, x, other));
            }
        }
        out
    }

    /// Generate the reduced table: breadth-first, in place with 2 bits per entry (3 = not
    /// reached yet), then every non-canonical stabiliser variant gets its canonical value.
    pub fn generate(st: &Staging, threads: usize, verbose: bool) -> PatTable {
        let t0 = std::time::Instant::now();
        let n = N_RED;
        let words = n.div_ceil(32);
        let pt = PatTable::new(
            Mod3Table {
                data: HugeBuf::zeroed(words),
            },
            false,
        );
        let w: &[AtomicU64] = unsafe {
            std::slice::from_raw_parts(pt.table.data.as_ptr() as *const AtomicU64, words)
        };
        crate::par::for_chunks(words, 1 << 16, |r| {
            for i in r {
                w[i].store(u64::MAX, Ordering::Relaxed);
            }
        });
        let get = |i: usize| (w[i >> 5].load(Ordering::Relaxed) >> ((i & 31) * 2) & 3) as u8;
        let set = |i: usize, code: u8| -> bool {
            let sh = (i & 31) * 2;
            let mask = !((3 ^ code as u64) << sh);
            let old = w[i >> 5].fetch_and(mask, Ordering::Relaxed);
            old >> sh & 3 == 3
        };
        let centre_idx: [V32; N_MOVES] = std::array::from_fn(|m| {
            let mut v = V32(std::array::from_fn(|i| i as u8));
            v.0[..24].copy_from_slice(&MOVE_PERMS[m].centre_src);
            v
        });
        let pt_ref = &pt;
        let mut count = 0usize;
        for g in Self::goal_flags() {
            if set(pt_ref.canonical(st, &g) as usize, 0) {
                count += 1;
            }
        }
        let mut frontier = count;
        let mut d = 0u8;
        while frontier > 0 {
            let forward = frontier < (n - count) / 8;
            let code = d % 3;
            let next_code = (d + 1) % 3;
            let new = AtomicUsize::new(0);
            let chunk = 1 << 14;
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut local = 0;
                        loop {
                            let start = next.fetch_add(chunk, Ordering::Relaxed);
                            if start >= n {
                                break;
                            }
                            for i in start..(start + chunk).min(n) {
                                let v = get(i);
                                if forward {
                                    if v != code {
                                        continue;
                                    }
                                    let g = pt_ref.red_state(st, i);
                                    for idx in &centre_idx {
                                        let j = pt_ref.canonical(st, &g.gather(idx)) as usize;
                                        if get(j) == 3 && set(j, next_code) {
                                            local += 1;
                                        }
                                    }
                                } else if v == 3 {
                                    let g = pt_ref.red_state(st, i);
                                    for idx in &centre_idx {
                                        let j = pt_ref.canonical(st, &g.gather(idx)) as usize;
                                        if get(j) == code {
                                            if set(i, next_code) {
                                                local += 1;
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        new.fetch_add(local, Ordering::Relaxed);
                    });
                }
            });
            frontier = new.load(Ordering::Relaxed);
            count += frontier;
            d += 1;
            if verbose {
                eprintln!(
                    "  pattern table depth {d}: {frontier} ({count} of {n}, {:.1} s)",
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // Every non-canonical entry gets its canonical entry's value.
        crate::par::for_chunks(n, 1 << 14, |r| {
            for i in r {
                let j = pt_ref.canonical(st, &pt_ref.red_state(st, i)) as usize;
                if j != i {
                    let c = get(j) as u64;
                    let sh = (i & 31) * 2;
                    if w[i >> 5].load(Ordering::Relaxed) >> sh & 3 != c {
                        w[i >> 5].fetch_and(!(3 << sh), Ordering::Relaxed);
                        w[i >> 5].fetch_or(c << sh, Ordering::Relaxed);
                    }
                }
            }
        });
        pt
    }

    /// The large table, filled from the reduced one.
    pub fn expand(&self, st: &Staging) -> PatTable {
        assert!(!self.big);
        let words = N_BIG.div_ceil(32);
        let mut data = HugeBuf::<u64>::zeroed(words);
        let out = data.as_mut_slice();
        let ptr = out.as_mut_ptr() as usize;
        crate::par::for_chunks(words, 1 << 12, |r| {
            let mut i = r.start * 32;
            let (mut rank, mut rem) = (i / N_XQ, i % N_XQ);
            let mut pinfo = (0u32, 0u32, u32::MAX);
            for wi in r {
                let mut word = 0u64;
                for j in 0..32 {
                    if i < N_BIG {
                        if pinfo.2 != rank as u32 {
                            let p = comb_unrank(rank as u32, 24, 8);
                            pinfo = (p, st.a_info[rank], rank as u32);
                        }
                        let p = pinfo.0;
                        let x = pdep(self.unl8_inv[rem / N_Q] as u32, p);
                        let q = pdep(st.unl_inv[rem % N_Q] as u32, !p & MASK24);
                        let v = self
                            .table
                            .get(self.red_index(st, pinfo.1, &masks_flags(p, x, q)));
                        word |= (v as u64) << (2 * j);
                    } else {
                        word |= 3 << (2 * j);
                    }
                    i += 1;
                    rem += 1;
                    if rem == N_XQ {
                        rem = 0;
                        rank += 1;
                    }
                }
                unsafe { *(ptr as *mut u64).add(wi) = word };
            }
        });
        PatTable::new(Mod3Table { data }, true)
    }
}

/// Load (or generate and save) the pattern table in `dir`: the reduced one, or the large one
/// (`big`), which is filled from the reduced one.
pub fn load(
    dir: &std::path::Path,
    st: &Staging,
    big: bool,
    threads: usize,
    verbose: bool,
) -> PatTable {
    let name = if big { "p1-pat-big" } else { "p1-pat" };
    let path = dir.join(format!("{name}.tab"));
    let words = PatTable::len(big).div_ceil(32);
    let mut table = Mod3Table {
        data: HugeBuf::zeroed(words),
    };
    if crate::table::load(&path, name, table.data.as_bytes_mut()).is_ok() {
        return PatTable::new(table, big);
    }
    drop(table);
    let red = {
        let rpath = dir.join("p1-pat.tab");
        let mut t = Mod3Table {
            data: HugeBuf::zeroed(N_RED.div_ceil(32)),
        };
        if crate::table::load(&rpath, "p1-pat", t.data.as_bytes_mut()).is_ok() {
            PatTable::new(t, false)
        } else {
            drop(t);
            if verbose {
                eprintln!("generating the phase-1 pattern table (0.9 GB)...");
            }
            crate::par::set_threads(threads);
            let t = PatTable::generate(st, threads, verbose);
            crate::table::save(&rpath, "p1-pat", t.table.data.as_bytes()).expect("saving table");
            t
        }
    };
    if !big {
        return red;
    }
    if verbose {
        eprintln!("filling the large phase-1 pattern table (41 GB)...");
    }
    let t = red.expand(st);
    drop(red);
    crate::table::save(&path, name, t.table.data.as_bytes()).expect("saving table");
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::Cube;
    use crate::phase1::centre_syms;

    /// The canonical index is the same for every symmetric image of a state, with the pair's
    /// colours swapped, and with the other two pairs swapped; the fast index is that of one
    /// of the state's stabiliser variants.
    #[test]
    fn canonical_index_is_symmetric() {
        let st = Staging::new();
        let pt = PatTable::new(Mod3Table::from_depths(&[0]), false);
        let syms = centre_syms();
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let s = P1State::from_cube(&Cube::random(&mut rng));
            for c in 0..3 {
                let g = pt.flags(&s.centres, c);
                let (p, x, q) = flag_masks(&g);
                let i = pt.canonical(&st, &g);
                let r = !(p | q) & MASK24;
                assert_eq!(pt.canonical(&st, &masks_flags(p, p & !x, q)), i);
                assert_eq!(pt.canonical(&st, &masks_flags(p, x, r)), i);
                for sy in &syms {
                    let img = |m: u32| -> u32 {
                        (0..24)
                            .filter(|&j| m >> j & 1 == 1)
                            .map(|j| 1 << sy[j])
                            .sum()
                    };
                    let h = masks_flags(img(p), img(x), img(q));
                    assert_eq!(pt.canonical(&st, &h), i);
                }
                let info = st.a_info[st.rank24(p) as usize];
                let j = pt.red_index(&st, info, &g);
                assert_eq!(pt.canonical(&st, &pt.red_state(&st, j as usize)), i);
            }
        }
    }
}
