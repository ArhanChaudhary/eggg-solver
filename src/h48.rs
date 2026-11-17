//! H48 pruning tables for the 3x3x3 (after Sebastiano Tronto's H48 solver and egg-solver).
//!
//! Target group: every corner in its own tetrad and oriented, every edge in its own slice
//! (and, for h11, oriented). The corner part (orientation + tetrad separation) is reduced by
//! the 48 symmetries to 3393 classes; the edge part is that of the cube conjugated into the
//! class representative. hK keeps K of the 11 edge-orientation bits; its entries are the
//! minimum over the h11 cosets they cover.
//!
//! Layout: rows of 512 two-bit slots (128 bytes), one per (class, msplit, top K eo bits),
//! indexed by the E-slice position (495 values). Slots 0-1 of every group of 64 hold the
//! 4-bit minimum of the group; values: code 0 = "at most base" (use the group minimum),
//! codes 1..3 = base + code (3 meaning at least base + 3).

use crate::comb::comb_rank;
use crate::cube::{Mat, ROTATIONS, mat_apply};
use crate::cube3::{Cube3, N_MOVES3};
use crate::moves::ALLOWED_AFTER;
use crate::table::HugeBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub const N_COCSEP: usize = 2187 * 70;
pub const N_CLASSES: usize = 3393;
pub const N_ESLOT: usize = 495;
pub const N_MSPLIT: usize = 70;

/// Corner positions/normals and edge positions/reference normals of the 3x3x3 (doubled
/// coordinates are not needed: unit coordinates).
const CPOS: [[i8; 3]; 8] = [
    [1, 1, 1],
    [-1, 1, 1],
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, -1, -1],
    [1, -1, -1],
];
const EPOS: [[i8; 3]; 12] = [
    [1, 1, 0],
    [0, 1, 1],
    [-1, 1, 0],
    [0, 1, -1],
    [1, -1, 0],
    [0, -1, 1],
    [-1, -1, 0],
    [0, -1, -1],
    [1, 0, 1],
    [-1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
];
/// Tetrad of each corner piece: 0 for {URF, ULB, DLF, DRB}, 1 for the others.
pub const TETRAD: [u8; 8] = [0, 1, 0, 1, 1, 0, 1, 0];
/// Slice of each edge piece: 0 = M (UF UB DF DB), 1 = S (UR UL DR DL), 2 = E.
pub const SLICE: [u8; 12] = [1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 2, 2];

fn corner_normals(i: usize) -> [[i8; 3]; 3] {
    // Same order as the 4x4x4 model: U/D sticker first, then clockwise.
    let g = &*crate::cube::GEO;
    g.corner_normals[i]
}

fn edge_ref(i: usize) -> [i8; 3] {
    let p = EPOS[i];
    if p[1] != 0 {
        [0, p[1], 0]
    } else {
        [0, 0, p[2]]
    }
}

/// The 48 symmetries as matrices (24 rotations, then each composed with -I).
pub fn sym_mats() -> Vec<Mat> {
    let mut v: Vec<Mat> = ROTATIONS.to_vec();
    for m in ROTATIONS.iter() {
        v.push(m.map(|r| r.map(|x| -x)));
    }
    v
}

/// Symmetry data for the 3x3x3.
pub struct Syms {
    /// Corner slot images.
    pub cslot: [[u8; 8]; 48],
    pub eslot: [[u8; 12]; 48],
    /// Edge flip contributions: eo'(s(i)) = eo(i) ^ a[s][piece] ^ b[s][i].
    pub ea: [[u8; 12]; 48],
    pub eb: [[u8; 12]; 48],
    pub inv: [u8; 48],
    /// mul[a][b]: the symmetry a * b (conjugating by b, then by a).
    pub mul: [[u8; 48]; 48],
    /// New orientation of a corner: cconj[k][slot][piece][ori].
    cconj: Vec<[[[u8; 3]; 8]; 8]>,
}

impl Syms {
    pub fn new() -> Syms {
        let mats = sym_mats();
        let mut s = Syms {
            cslot: [[0; 8]; 48],
            eslot: [[0; 12]; 48],
            ea: [[0; 12]; 48],
            eb: [[0; 12]; 48],
            inv: [0; 48],
            mul: [[0; 48]; 48],
            cconj: vec![[[[0; 3]; 8]; 8]; 48],
        };
        for (k, m) in mats.iter().enumerate() {
            for i in 0..8 {
                let p = mat_apply(m, CPOS[i]);
                s.cslot[k][i] = CPOS.iter().position(|&q| q == p).unwrap() as u8;
            }
            for i in 0..12 {
                let p = mat_apply(m, EPOS[i]);
                let j = EPOS.iter().position(|&q| q == p).unwrap();
                s.eslot[k][i] = j as u8;
                let flip = (mat_apply(m, edge_ref(i)) != edge_ref(j)) as u8;
                s.ea[k][i] = flip;
                s.eb[k][i] = flip;
            }
            let mut inv = 0;
            for (l, n) in mats.iter().enumerate() {
                if crate::cube::mat_mul(m, n) == crate::cube::IDENTITY {
                    inv = l;
                }
            }
            s.inv[k] = inv as u8;
            for (l, n) in mats.iter().enumerate() {
                let prod = crate::cube::mat_mul(m, n);
                s.mul[k][l] = mats.iter().position(|q| *q == prod).unwrap() as u8;
            }
        }
        for k in 0..48 {
            let m = &mats[k];
            for i in 0..8 {
                for p in 0..8 {
                    for o in 0..3 {
                        let ni = s.cslot[k][i] as usize;
                        let np = s.cslot[k][p] as usize;
                        // The piece's sticker that becomes the new piece's reference colour.
                        let ref_new = corner_normals(np)[0];
                        let kk = (0..3)
                            .find(|&q| mat_apply(m, corner_normals(p)[q]) == ref_new)
                            .unwrap();
                        // That sticker sits on slot i's sticker (kk + o) % 3.
                        let at = mat_apply(m, corner_normals(i)[(kk + o) % 3]);
                        s.cconj[k][i][p][o] =
                            corner_normals(ni).iter().position(|&n| n == at).unwrap() as u8;
                    }
                }
            }
        }
        s
    }

    /// The conjugate s X s^-1 (pieces and slots moved by the symmetry, colours relabelled).
    pub fn conj(&self, x: &Cube3, k: usize) -> Cube3 {
        let mut y = *x;
        for i in 0..8 {
            let p = x.cp[i] as usize;
            let ni = self.cslot[k][i] as usize;
            y.cp[ni] = self.cslot[k][p];
            y.co[ni] = self.cconj[k][i][p][x.co[i] as usize];
        }
        for i in 0..12 {
            let p = x.ep[i] as usize;
            let ni = self.eslot[k][i] as usize;
            y.ep[ni] = self.eslot[k][p];
            y.eo[ni] = x.eo[i] ^ self.ea[k][p] ^ self.eb[k][i];
        }
        y
    }
}

impl Default for Syms {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
pub fn co_index(co: &[u8; 8]) -> usize {
    co[..7].iter().rev().fold(0, |a, &x| a * 3 + x as usize)
}

pub fn co_unindex(mut i: usize) -> [u8; 8] {
    let mut co = [0u8; 8];
    let mut s = 0;
    for c in co.iter_mut().take(7) {
        *c = (i % 3) as u8;
        s += *c;
        i /= 3;
    }
    co[7] = (3 - s % 3) % 3;
    co
}

/// Static coordinate data.
pub struct Coords {
    pub syms: Syms,
    /// cocsep -> class << 16 | ttrep << 8 | corner distance (to tetrads + oriented).
    pub cocsep: Vec<u32>,
    pub class_rep: Vec<u32>,
    /// Symmetries (other than the identity) fixing the class representative's cocsep.
    pub stab: Vec<Vec<u8>>,
    rank70: [u8; 256],
    unrank70: [u8; 70],
    rank495: Vec<u16>,
    unrank495: Vec<u16>,
}

fn sep_mask(cp: &[u8; 8]) -> u8 {
    let mut m = 0;
    for i in 0..8 {
        m |= TETRAD[cp[i] as usize] << i;
    }
    m
}

impl Coords {
    pub fn new() -> Coords {
        let syms = Syms::new();
        let mut rank70 = [255u8; 256];
        let mut unrank70 = [0u8; 70];
        for m in 0u32..256 {
            if m.count_ones() == 4 {
                let r = comb_rank(m) as u8;
                rank70[m as usize] = r;
                unrank70[r as usize] = m as u8;
            }
        }
        let mut rank495 = vec![u16::MAX; 4096];
        let mut unrank495 = vec![0u16; N_ESLOT];
        for m in 0u32..4096 {
            if m.count_ones() == 4 {
                let r = comb_rank(m) as u16;
                rank495[m as usize] = r;
                unrank495[r as usize] = m as u16;
            }
        }
        let mut c = Coords {
            syms,
            cocsep: vec![u32::MAX; N_COCSEP],
            class_rep: vec![],
            stab: vec![],
            rank70,
            unrank70,
            rank495,
            unrank495,
        };
        // Classes.
        for idx in 0..N_COCSEP {
            if c.cocsep[idx] != u32::MAX {
                continue;
            }
            let x = c.cocsep_cube(idx);
            let class = c.class_rep.len() as u32;
            c.class_rep.push(idx as u32);
            let mut st = vec![];
            for k in 0..48 {
                let y = c.syms.conj(&x, k);
                let j = c.cocsep_of(&y);
                if j == idx && k != 0 {
                    st.push(k as u8);
                }
                if c.cocsep[j] == u32::MAX {
                    // ttrep: the symmetry taking j to the representative is the inverse of k.
                    c.cocsep[j] = class << 16 | (c.syms.inv[k] as u32) << 8;
                }
            }
            c.stab.push(st);
        }
        assert_eq!(c.class_rep.len(), N_CLASSES);
        // Corner distance to the target (BFS over cocsep).
        let mut dist = vec![u8::MAX; N_COCSEP];
        let goal = c.cocsep_of(&Cube3::SOLVED);
        dist[goal] = 0;
        let mut frontier = vec![goal];
        let mut d = 0;
        while !frontier.is_empty() {
            let mut next = vec![];
            for &i in &frontier {
                let x = c.cocsep_cube(i);
                for m in 0..N_MOVES3 {
                    let j = c.cocsep_of(&x.apply(m));
                    if dist[j] == u8::MAX {
                        dist[j] = d + 1;
                        next.push(j);
                    }
                }
            }
            frontier = next;
            d += 1;
        }
        for i in 0..N_COCSEP {
            c.cocsep[i] |= dist[i] as u32;
        }
        c
    }

    pub fn cocsep_of(&self, x: &Cube3) -> usize {
        co_index(&x.co) * 70 + self.rank70[sep_mask(&x.cp) as usize] as usize
    }

    /// A corner state with the given cocsep (edges solved).
    pub fn cocsep_cube(&self, idx: usize) -> Cube3 {
        let mut x = Cube3::SOLVED;
        x.co = co_unindex(idx / 70);
        let sep = self.unrank70[idx % 70];
        let (mut a, mut b) = (0usize, 0usize);
        let ta: Vec<u8> = (0..8)
            .filter(|&p| TETRAD[p] == 0)
            .map(|p| p as u8)
            .collect();
        let tb: Vec<u8> = (0..8)
            .filter(|&p| TETRAD[p] == 1)
            .map(|p| p as u8)
            .collect();
        for i in 0..8 {
            if sep >> i & 1 == 1 {
                x.cp[i] = tb[b];
                b += 1;
            } else {
                x.cp[i] = ta[a];
                a += 1;
            }
        }
        x
    }

    /// Edge part (eslot, msplit, eo11) of a cube.
    #[inline]
    #[cfg(test)]
    pub fn edge_part(&self, x: &Cube3) -> (u32, u32, u32) {
        let mut emask = 0u32;
        let mut mmask = 0u32;
        let mut eo = 0u32;
        for i in 0..12 {
            let s = SLICE[x.ep[i] as usize];
            emask |= ((s == 2) as u32) << i;
            mmask |= ((s == 0) as u32) << i;
            eo |= (x.eo[i] as u32) << i;
        }
        let eslot = self.rank495[emask as usize] as u32;
        let msplit = self.rank70[crate::phase1::pext(mmask, !emask & 0xfff) as usize] as u32;
        (eslot, msplit, eo & 0x7ff)
    }

    /// Edge part of the cube conjugated by symmetry k (computed on the edges only).
    #[inline]
    pub fn edge_part_conj(&self, x: &Cube3, k: usize) -> (u32, u32, u32) {
        let mut emask = 0u32;
        let mut mmask = 0u32;
        let mut eo = 0u32;
        let sl = &self.syms.eslot[k];
        for i in 0..12 {
            let p = x.ep[i] as usize;
            let j = sl[i] as u32;
            let s = SLICE[sl[p] as usize];
            emask |= ((s == 2) as u32) << j;
            mmask |= ((s == 0) as u32) << j;
            eo |= ((x.eo[i] ^ self.syms.ea[k][p] ^ self.syms.eb[k][i]) as u32) << j;
        }
        let eslot = self.rank495[emask as usize] as u32;
        let msplit = self.rank70[crate::phase1::pext(mmask, !emask & 0xfff) as usize] as u32;
        (eslot, msplit, eo & 0x7ff)
    }

    /// The h11 keys of a cube: its class, and the edge part conjugated into the class
    /// representative by every symmetry that does so (one for most classes; all of them must
    /// be marked so that any of them may be used for lookups).
    pub fn h11_keys(&self, x: &Cube3, out: &mut Vec<u64>) {
        out.clear();
        let cs = self.cocsep[self.cocsep_of(x)];
        let class = (cs >> 16) as u64;
        let tt = (cs >> 8 & 255) as usize;
        let key =
            |e: (u32, u32, u32)| class << 27 | (e.1 as u64) << 20 | (e.2 as u64) << 9 | e.0 as u64;
        out.push(key(self.edge_part_conj(x, tt)));
        let st = &self.stab[class as usize];
        if !st.is_empty() {
            for &t in st {
                out.push(key(
                    self.edge_part_conj(x, self.syms.mul[t as usize][tt] as usize)
                ));
            }
            out.sort_unstable();
            out.dedup();
        }
    }

    /// A representative cube of an h11 key.
    pub fn key_cube(&self, key: u64) -> Cube3 {
        let class = (key >> 27) as usize;
        let msplit = (key >> 20 & 127) as usize;
        let eo = (key >> 9 & 0x7ff) as u32;
        let eslot = (key & 511) as usize;
        let mut x = self.cocsep_cube(self.class_rep[class] as usize);
        let emask = self.unrank495[eslot] as u32;
        let rest = !emask & 0xfff;
        let mmask = crate::phase1::pdep(self.unrank70[msplit] as u32, rest);
        let (mut ie, mut im, mut is) = (0, 0, 0);
        let pe = [8u8, 9, 10, 11];
        let pm = [1u8, 3, 5, 7];
        let ps = [0u8, 2, 4, 6];
        let mut sum = 0;
        for i in 0..12 {
            if emask >> i & 1 == 1 {
                x.ep[i] = pe[ie];
                ie += 1;
            } else if mmask >> i & 1 == 1 {
                x.ep[i] = pm[im];
                im += 1;
            } else {
                x.ep[i] = ps[is];
                is += 1;
            }
            if i < 11 {
                x.eo[i] = (eo >> i & 1) as u8;
                sum += x.eo[i];
            }
        }
        x.eo[11] = sum % 2;
        x
    }
}

impl Default for Coords {
    fn default() -> Self {
        Self::new()
    }
}

/// Physical slot of an E-slice rank in a 512-slot row (slots 0-1 of each 64 hold the minimum).
#[inline(always)]
pub fn slot_of(eslot: u32) -> u32 {
    (eslot / 62) * 64 + 2 + eslot % 62
}

pub const DEFAULT_BASE: [u8; 12] = [7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10];

/// An hK table.
pub struct H48 {
    pub k: u32,
    pub base: u8,
    pub data: HugeBuf<u64>,
}

impl H48 {
    pub fn rows(k: u32) -> usize {
        (N_CLASSES * N_MSPLIT) << k
    }

    pub fn bytes(k: u32) -> usize {
        Self::rows(k) * 128
    }

    #[inline(always)]
    pub fn index(&self, class: u32, eslot: u32, msplit: u32, eo11: u32) -> usize {
        let row = ((class * N_MSPLIT as u32 + msplit) << self.k | eo11 >> (11 - self.k)) as usize;
        row * 512 + slot_of(eslot) as usize
    }

    #[inline(always)]
    pub fn code(&self, i: usize) -> u8 {
        (self.data[i >> 5] >> ((i & 31) * 2) & 3) as u8
    }

    /// The 4-bit minimum of the group containing slot i.
    #[inline(always)]
    pub fn group_min(&self, i: usize) -> u8 {
        (self.data[(i & !63) >> 5] & 15) as u8
    }

    #[inline(always)]
    pub fn prefetch(&self, i: usize) {
        crate::table::prefetch(unsafe { self.data.as_ptr().add(i >> 5) });
    }

    /// Lower bound from entry `i`.
    #[inline(always)]
    pub fn value(&self, i: usize) -> u8 {
        let c = self.code(i);
        if c == 0 {
            self.group_min(i)
        } else {
            self.base + c
        }
    }

    /// Generate the table: BFS over h11 cosets to `prefix` moves, then depth-first passes
    /// from the cosets at that depth up to `base + 2` moves.
    pub fn generate(co: &Coords, k: u32, base: u8, threads: usize, verbose: bool) -> H48 {
        let t0 = std::time::Instant::now();
        let n_words = Self::bytes(k) / 8;
        let data = HugeBuf::<u64>::zeroed(n_words);
        // Start with every code = 3 and every group minimum = 15.
        let words: &[AtomicU64] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const AtomicU64, n_words) };
        crate::par::for_chunks(n_words, 1 << 16, |r| {
            for w in r {
                words[w].store(u64::MAX, Ordering::Relaxed);
            }
        });
        let table = H48 { k, base, data };
        let mark = |key: u64, depth: u8| {
            let class = (key >> 27) as u32;
            let msplit = (key >> 20 & 127) as u32;
            let eo = (key >> 9 & 0x7ff) as u32;
            let eslot = (key & 511) as u32;
            let i = table.index(class, eslot, msplit, eo);
            let code = depth.saturating_sub(base).min(3) as u64;
            let w = &words[i >> 5];
            let sh = (i & 31) * 2;
            // Lower the 2-bit code to `code` if larger.
            let cur = w.load(Ordering::Relaxed) >> sh & 3;
            if cur > code {
                w.fetch_and(!((cur ^ code) << sh), Ordering::Relaxed);
            }
            if depth <= base {
                let gw = &words[(i & !63) >> 5];
                let mut old = gw.load(Ordering::Relaxed);
                while (old & 15) as u8 > depth {
                    let new = (old & !15) | depth as u64;
                    match gw.compare_exchange_weak(old, new, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(o) => old = o,
                    }
                }
            }
        };
        // Prefix BFS over canonical h11 keys.
        let prefix: u8 = 8.min(base + 2);
        let mut levels: Vec<Vec<u64>> = vec![];
        let mut keys = vec![];
        co.h11_keys(&Cube3::SOLVED, &mut keys);
        levels.push(vec![keys[0]]);
        let mut seen: std::collections::HashSet<u64> = levels[0].iter().copied().collect();
        for d in 0..prefix {
            let cur = &levels[d as usize];
            let next = std::sync::Mutex::new(Vec::new());
            let idx = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut local = vec![];
                        let mut ks = vec![];
                        loop {
                            let i = idx.fetch_add(1024, Ordering::Relaxed);
                            if i >= cur.len() {
                                break;
                            }
                            for &key in &cur[i..(i + 1024).min(cur.len())] {
                                let x = co.key_cube(key);
                                for m in 0..N_MOVES3 {
                                    co.h11_keys(&x.apply(m), &mut ks);
                                    local.push(ks[0]);
                                }
                            }
                        }
                        next.lock().unwrap().extend(local);
                    });
                }
            });
            let mut nx = next.into_inner().unwrap();
            nx.sort_unstable();
            nx.dedup();
            nx.retain(|k| !seen.contains(k));
            seen.extend(nx.iter().copied());
            if verbose {
                eprintln!(
                    "  h48 prefix depth {}: {} cosets ({:.1} s)",
                    d + 1,
                    nx.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
            levels.push(nx);
        }
        drop(seen);
        // Mark the prefix (every stabiliser variant of every coset).
        for (d, lv) in levels.iter().enumerate() {
            crate::par::for_chunks(lv.len(), 4096, |r| {
                let mut ks = vec![];
                for &key in &lv[r] {
                    co.h11_keys(&co.key_cube(key), &mut ks);
                    for &k2 in &ks {
                        mark(k2, d as u8);
                    }
                }
            });
        }
        // Depth-first passes from the last prefix level.
        let roots = &levels[prefix as usize];
        for len in prefix + 1..=base + 2 {
            let idx = AtomicUsize::new(0);
            let leaves = AtomicU64::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut ks = vec![];
                        let mut n = 0u64;
                        loop {
                            let i = idx.fetch_add(64, Ordering::Relaxed);
                            if i >= roots.len() {
                                break;
                            }
                            for &key in &roots[i..(i + 64).min(roots.len())] {
                                let x = co.key_cube(key);
                                dfs_mark(
                                    co,
                                    &x,
                                    len - prefix,
                                    crate::moves::N_MOVES,
                                    len,
                                    &mut ks,
                                    &mut n,
                                    &mark,
                                );
                            }
                        }
                        leaves.fetch_add(n, Ordering::Relaxed);
                    });
                }
            });
            if verbose {
                eprintln!(
                    "  h48 pass depth {len}: {} leaves ({:.1} s)",
                    leaves.load(Ordering::Relaxed),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // Groups whose minimum is still 15 have no entry at most base: set base + 1.
        crate::par::for_chunks(n_words / 2, 1 << 14, |r| {
            for g in r {
                let w = &words[g * 2];
                let v = w.load(Ordering::Relaxed);
                if v & 15 == 15 {
                    w.store((v & !15) | (base as u64 + 1).min(15), Ordering::Relaxed);
                }
            }
        });
        table
    }
}

#[allow(clippy::too_many_arguments)]
fn dfs_mark(
    co: &Coords,
    x: &Cube3,
    left: u8,
    last: usize,
    depth: u8,
    ks: &mut Vec<u64>,
    n: &mut u64,
    mark: &(impl Fn(u64, u8) + Sync),
) {
    if left == 0 {
        *n += 1;
        co.h11_keys(x, ks);
        for &k in ks.iter() {
            mark(k, depth);
        }
        return;
    }
    let mut allowed = ALLOWED_AFTER[last] & ((1 << N_MOVES3) - 1);
    while allowed != 0 {
        let m = allowed.trailing_zeros() as usize;
        allowed &= allowed - 1;
        dfs_mark(co, &x.apply(m), left - 1, m, depth, ks, n, mark);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random3(rng: &mut impl Rng) -> Cube3 {
        let mut x = Cube3::SOLVED;
        for _ in 0..40 {
            x = x.apply(rng.gen_range(0..18));
        }
        x
    }

    #[test]
    fn conjugation_is_a_homomorphism() {
        let s = Syms::new();
        let mut rng = rand::thread_rng();
        for k in 0..48 {
            let x = random3(&mut rng);
            // conj(x * m) = conj(x) * conj(m) where conj(m) is the conjugated move.
            let y = s.conj(&x, k);
            assert!(y.is_valid(), "sym {k}");
            assert_eq!(s.conj(&y, s.inv[k] as usize), x);
            assert_eq!(s.conj(&Cube3::SOLVED, k), Cube3::SOLVED);
        }
    }

    #[test]
    fn conjugation_maps_moves_to_moves() {
        let s = Syms::new();
        let mut rng = rand::thread_rng();
        for k in 0..48 {
            for m in 0..18 {
                let cm = s.conj(&Cube3::SOLVED.apply(m), k);
                let m2 = (0..18)
                    .find(|&j| Cube3::SOLVED.apply(j) == cm)
                    .expect("conjugate of a move is a move");
                let x = random3(&mut rng);
                assert_eq!(
                    s.conj(&x.apply(m), k),
                    s.conj(&x, k).apply(m2),
                    "sym {k} move {m}"
                );
            }
        }
    }

    #[test]
    fn edge_part_conj_matches() {
        let c = Coords::new();
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let x = random3(&mut rng);
            for k in 0..48 {
                assert_eq!(c.edge_part_conj(&x, k), c.edge_part(&c.syms.conj(&x, k)));
            }
        }
    }

    #[test]
    fn cocsep_is_well_defined() {
        // The class of a cube's corner part does not depend on the corner permutation within
        // the tetrads.
        let c = Coords::new();
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let x = random3(&mut rng);
            let i = c.cocsep_of(&x);
            for k in 0..48 {
                let a = c.cocsep_of(&c.syms.conj(&x, k));
                let b = c.cocsep_of(&c.syms.conj(&c.cocsep_cube(i), k));
                assert_eq!(a, b);
            }
        }
    }
}
