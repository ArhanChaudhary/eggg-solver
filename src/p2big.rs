//! A large joint phase-2 pruning table over the pairing permutation tau, the U/D *and* F/B
//! centre patterns and the PLL-parity bit: everything of the phase-2 state but the R/L centre
//! pattern (12 values, covered by the exact centre table), so its bound is close to the exact
//! phase-2 distance.
//!
//! Size: 70 * 70 * 12!/2 * 2 = 2.35e12 states, reduced by 32 symmetries: the spatial
//! symmetries (rotations and reflections) that keep the R/L axis and map the low wing slots to
//! low ones, times the colour swaps U<->D and F<->B (the goal has every orientation of the U/D
//! and F/B centres once the R/L ones are ignored). The pair of patterns is reduced first, then
//! the M-slice entries of tau under the pair's stabiliser; the other eight entries are ranked
//! (8!/2). Distances are stored mod 3 in 2 bits (~18 GB).

use crate::cube::{EDGE_POS, FACE_NORMAL, GEO, Mat, centre_slot, face_of_normal, mat_apply};
use crate::p2joint::{
    M_EDGES, N_PREFIX, N_REST, Sym, WIDE, conj, conj_edges, perm8_unrank, prefix_rank,
    prefix_unrank, rest_rank,
};
use crate::phase2::P2Tables;
use crate::simd::V32;
use crate::table::HugeBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const N_PAIR: usize = 70 * 70;
const UD_POS: [usize; 8] = [0, 1, 2, 3, 12, 13, 14, 15];
const FB_POS: [usize; 8] = [8, 9, 10, 11, 20, 21, 22, 23];

struct BSym {
    /// Edge part (conjugation of tau).
    sym: Sym,
    /// The edge part is the identity (a pure colour swap).
    plain: bool,
    /// Image of a pattern pair ud * 70 + fb.
    pair: Vec<u16>,
}

pub struct Big {
    syms: Vec<BSym>,
    /// Pattern pair -> pair class << 5 | symmetry taking the pair to the class representative.
    pair_cls: Vec<u32>,
    /// (pair class, M-slice prefix) -> class << 5 | symmetry (in the pair's stabiliser) taking
    /// the prefix to the class representative.
    pref_cls: Vec<u32>,
    /// Class -> representative (pair, prefix).
    rep: Vec<(u16, u16)>,
    /// Class -> stabiliser of the representative (bit s: symmetry s fixes it), identity
    /// excluded.
    stab: Vec<u32>,
    pub fb_move: [[u8; 17]; 70],
    pub ud_move: [[u8; 17]; 70],
    pub table: HugeBuf<u64>,
    pub n: usize,
}

/// The 48 spatial symmetries of the cube (rotations and reflections).
fn all_mats() -> Vec<Mat> {
    let mut out = vec![];
    for p in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        for signs in 0..8 {
            let mut m = [[0i8; 3]; 3];
            for i in 0..3 {
                m[i][p[i]] = if signs >> i & 1 == 1 { -1 } else { 1 };
            }
            out.push(m);
        }
    }
    out
}

impl Big {
    /// Symmetries, classes and move tables (no table data).
    pub fn new(p2: &P2Tables) -> Big {
        let g = &*GEO;
        let mut geo: Vec<(Mat, [u8; 12])> = vec![];
        for m in all_mats() {
            if m[0][0] == 0 {
                continue;
            }
            let mut e = [0u8; 12];
            let mut lows = 0;
            for i in 0..12 {
                let p = mat_apply(&m, EDGE_POS[i]);
                e[i] = EDGE_POS.iter().position(|&q| q == p).unwrap() as u8;
                let img = crate::cube::wing_slot(mat_apply(&m, g.wing_pos[p2.low[i] as usize]));
                if img as u8 == p2.low[e[i] as usize] {
                    lows += 1;
                }
            }
            assert!(
                lows == 0 || lows == 12,
                "a symmetry mixes low and high slots"
            );
            if lows == 12 {
                geo.push((m, e));
            }
        }
        // Identity first.
        geo.sort_by_key(|(m, _)| *m != crate::cube::IDENTITY);
        assert_eq!(geo.len(), 8, "low-keeping symmetries of the R/L axis");
        let mut syms = vec![];
        for (m, e) in &geo {
            // Pattern pair images: a centre of colour c at position p goes to position m(p)
            // with colour m(c).
            let col = |f: usize| face_of_normal(mat_apply(m, FACE_NORMAL[f]));
            let mut base = vec![0u16; N_PAIR];
            for (pair, b) in base.iter_mut().enumerate() {
                let (ud, fb) = (
                    p2.unrank70((pair / 70) as u8),
                    p2.unrank70((pair % 70) as u8),
                );
                let (mut ud2, mut fb2) = (0u8, 0u8);
                for (pos, mask, c1, c0) in [
                    (&UD_POS, ud, crate::cube::U, crate::cube::D),
                    (&FB_POS, fb, crate::cube::F, crate::cube::B),
                ] {
                    for (i, &p) in pos.iter().enumerate() {
                        let c = if mask >> i & 1 == 1 { c1 } else { c0 };
                        let q = centre_slot(mat_apply(m, g.centre_pos[p]));
                        let c2 = col(c);
                        if let Some(j) = UD_POS.iter().position(|&x| x == q) {
                            if c2 == crate::cube::U {
                                ud2 |= 1 << j;
                            }
                        } else {
                            let j = FB_POS.iter().position(|&x| x == q).unwrap();
                            if c2 == crate::cube::F {
                                fb2 |= 1 << j;
                            }
                        }
                    }
                }
                *b = p2.rank70(ud2) as u16 * 70 + p2.rank70(fb2) as u16;
            }
            let (mut e_inv, mut e_map) = (V32([0x80; 32]), V32([0; 32]));
            for i in 0..12 {
                e_inv.0[e[i] as usize] = i as u8;
                e_map.0[i] = e[i];
            }
            let plain = *m == crate::cube::IDENTITY;
            for comp in 0..4u32 {
                let pair = base
                    .iter()
                    .map(|&x| {
                        let (mut ud, mut fb) = (x as usize / 70, x as usize % 70);
                        if comp & 1 == 1 {
                            ud = p2.rank70(!p2.unrank70(ud as u8)) as usize;
                        }
                        if comp & 2 == 2 {
                            fb = p2.rank70(!p2.unrank70(fb as u8)) as usize;
                        }
                        (ud * 70 + fb) as u16
                    })
                    .collect();
                syms.push(BSym {
                    sym: Sym {
                        e: *e,
                        ud: [0; 70],
                        e_inv,
                        e_map,
                    },
                    plain,
                    pair,
                });
            }
        }
        // Inverses (the group is not abelian: the rotoreflections about the R/L axis, which
        // swap the U/D and F/B axes, have order 4).
        let inv: Vec<u32> = syms
            .iter()
            .map(|s| {
                syms.iter()
                    .position(|t| {
                        (0..N_PAIR).all(|p| t.pair[s.pair[p] as usize] as usize == p)
                            && (0..12).all(|i| t.sym.e[s.sym.e[i] as usize] as usize == i)
                    })
                    .expect("symmetry without an inverse") as u32
            })
            .collect();
        let ns = syms.len();
        // Pair classes.
        let mut pair_cls = vec![u32::MAX; N_PAIR];
        let mut pair_rep = vec![];
        let mut pair_stab = vec![];
        for p in 0..N_PAIR {
            if pair_cls[p] != u32::MAX {
                continue;
            }
            let c = pair_rep.len() as u32;
            pair_rep.push(p);
            let mut st = 1u32;
            for (k, s) in syms.iter().enumerate() {
                let img = s.pair[p] as usize;
                if img == p {
                    st |= 1 << k;
                }
                if pair_cls[img] == u32::MAX {
                    pair_cls[img] = c << 5 | inv[k];
                }
            }
            pair_stab.push(st);
        }
        // Prefix classes within each pair class.
        let mut pref_cls = vec![u32::MAX; pair_rep.len() * N_PREFIX];
        let mut rep = vec![];
        let mut stab = vec![];
        for (c1, &p) in pair_rep.iter().enumerate() {
            let st1 = pair_stab[c1];
            for pr in 0..N_PREFIX {
                if pref_cls[c1 * N_PREFIX + pr] != u32::MAX {
                    continue;
                }
                let class = rep.len() as u32;
                rep.push((p as u16, pr as u16));
                let pv = prefix_unrank(pr);
                let mut st = 0u32;
                for (k, s) in syms.iter().enumerate() {
                    if st1 >> k & 1 == 0 {
                        continue;
                    }
                    let mut t = [0u8; 16];
                    for (i, &e) in M_EDGES.iter().enumerate() {
                        t[s.sym.e[e] as usize] = s.sym.e[pv[i] as usize];
                    }
                    let img = prefix_rank(&t);
                    if img == pr && k != 0 {
                        st |= 1 << k;
                    }
                    let slot = &mut pref_cls[c1 * N_PREFIX + img];
                    if *slot == u32::MAX {
                        *slot = class << 5 | inv[k];
                    }
                }
                stab.push(st);
            }
        }
        assert!(ns <= 32 && rep.len() < 1 << 27);
        let ud_move = {
            let mut t = [[0u8; 17]; 70];
            for (r, row) in t.iter_mut().enumerate() {
                for (i, x) in row.iter_mut().enumerate() {
                    *x = (p2.centre_move[r * 1680][i] / 1680) as u8;
                }
            }
            t
        };
        let fb_move = {
            let mut t = [[0u8; 17]; 70];
            for (r, row) in t.iter_mut().enumerate() {
                for (i, x) in row.iter_mut().enumerate() {
                    *x = (p2.centre_move[r * 24][i] / 24 % 70) as u8;
                }
            }
            t
        };
        let n = rep.len() * N_REST * 2;
        Big {
            syms,
            pair_cls,
            pref_cls,
            rep,
            stab,
            fb_move,
            ud_move,
            table: HugeBuf::zeroed(1),
            n,
        }
    }

    pub fn classes(&self) -> usize {
        self.rep.len()
    }

    pub fn bytes(&self) -> usize {
        self.n.div_ceil(32) * 8
    }

    /// The class and the tau conjugated into its representative's frame (one stabiliser
    /// variant).
    #[inline(always)]
    fn class_of(&self, pair: usize, tau: &[u8; 16]) -> (usize, [u8; 16]) {
        let c = self.pair_cls[pair];
        let s1 = &self.syms[(c & 31) as usize];
        let t1 = if s1.plain {
            *tau
        } else {
            conj_edges(tau, &s1.sym)
        };
        let k = self.pref_cls[(c >> 5) as usize * N_PREFIX + prefix_rank(&t1)];
        let s2 = &self.syms[(k & 31) as usize];
        let t2 = if s2.plain {
            t1
        } else {
            conj_edges(&t1, &s2.sym)
        };
        ((k >> 5) as usize, t2)
    }

    /// Table index from the full phase-2 centre coordinate and tau (one stabiliser variant;
    /// the table holds the same value in all of them).
    #[inline(always)]
    pub fn index(&self, cc: u32, tau: &[u8; 16]) -> usize {
        let pair = (cc / 24) as usize;
        let (class, t) = self.class_of(pair, tau);
        (class * N_REST + rest_rank(&t)) * 2 + (cc & 1) as usize
    }

    /// The canonical index (minimum over the stabiliser), used by the generator.
    fn canonical(&self, pair: usize, tau: &[u8; 16], pll: u32) -> usize {
        let (class, t) = self.class_of(pair, tau);
        let mut r = rest_rank(&t);
        let st = self.stab[class];
        if st != 0 {
            for k in 1..self.syms.len() {
                if st >> k & 1 == 1 {
                    r = r.min(rest_rank(&conj(&t, &self.syms[k].sym)));
                }
            }
        }
        (class * N_REST + r) * 2 + pll as usize
    }

    /// A state (pattern pair, tau, PLL bit) with the given index.
    fn state(&self, idx: usize) -> (usize, [u8; 16], u32) {
        let pll = (idx & 1) as u32;
        let r = idx / 2 % N_REST;
        let class = idx / 2 / N_REST;
        let (pair, pr) = self.rep[class];
        let pv = prefix_unrank(pr as usize);
        let mut tau = [0u8; 16];
        for (i, t) in tau.iter_mut().enumerate().skip(12) {
            *t = i as u8;
        }
        let mut pm = 0u32;
        for (i, &e) in M_EDGES.iter().enumerate() {
            tau[e] = pv[i];
            pm |= 1 << pv[i];
        }
        let mut avail = [0u8; 8];
        let mut k = 0;
        for v in 0..12u8 {
            if pm >> v & 1 == 0 {
                avail[k] = v;
                k += 1;
            }
        }
        let rest = crate::p2joint::REST;
        let mut p = perm8_unrank(r * 2);
        let mut full = tau;
        for (i, &e) in rest.iter().enumerate() {
            full[e] = avail[p[i] as usize];
        }
        if crate::comb::perm_parity(&full[..12]) == 1 {
            p.swap(6, 7);
            for (i, &e) in rest.iter().enumerate() {
                full[e] = avail[p[i] as usize];
            }
        }
        (pair as usize, full, pll)
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> u8 {
        (self.table[i >> 5] >> ((i & 31) * 2) & 3) as u8
    }

    #[inline(always)]
    pub fn prefetch(&self, i: usize) {
        crate::table::prefetch(unsafe { self.table.as_ptr().add(i >> 5) });
    }

    /// Exact distance of a state (full centre coordinate `cc`), by walking down the table.
    pub fn distance(&self, p2: &P2Tables, cc: u32, tau: &[u8; 16]) -> u8 {
        let (mut cc, mut tau) = (cc, *tau);
        let mut d = 0;
        loop {
            let v = self.get(self.index(cc, &tau));
            let next = (0..17)
                .map(|i| (p2.centre_move[cc as usize][i], p2.tau_move(&tau, i)))
                .find(|(c2, t2)| self.get(self.index(*c2, t2)) == (v + 2) % 3);
            match next {
                Some((c2, t2)) if d < 40 => {
                    cc = c2;
                    tau = t2;
                    d += 1;
                }
                _ => return d,
            }
        }
    }

    #[inline(always)]
    fn neighbour(&self, p2: &P2Tables, pair: usize, tau: &[u8; 16], pll: u32, k: usize) -> usize {
        let (ud, fb) = (pair / 70, pair % 70);
        let pair2 = self.ud_move[ud][k] as usize * 70 + self.fb_move[fb][k] as usize;
        self.canonical(pair2, &p2.tau_move(tau, k), pll ^ (WIDE >> k & 1))
    }

    /// The 17 neighbours' canonical indices of entry `idx`, xored (to time the generator's
    /// work per entry).
    #[doc(hidden)]
    pub fn probe(&self, p2: &P2Tables, idx: usize) -> usize {
        let (pair, tau, pll) = self.state(idx);
        (0..17).fold(0, |a, k| a ^ self.neighbour(p2, pair, &tau, pll, k))
    }

    /// Breadth-first generation, in place with 2 bits per entry (3 = not reached yet).
    pub fn generate(&mut self, p2: &P2Tables, threads: usize, verbose: bool) {
        let t0 = std::time::Instant::now();
        let n = self.n;
        let words = n.div_ceil(32);
        self.table = HugeBuf::zeroed(words);
        let w: &[AtomicU64] =
            unsafe { std::slice::from_raw_parts(self.table.as_ptr() as *const AtomicU64, words) };
        crate::par::for_chunks(words, 1 << 16, |r| {
            for i in r {
                w[i].store(u64::MAX, Ordering::Relaxed);
            }
        });
        self.table.collapse();
        let get = |i: usize| (w[i >> 5].load(Ordering::Relaxed) >> ((i & 31) * 2) & 3) as u8;
        let set = |i: usize, code: u8| -> bool {
            let sh = (i & 31) * 2;
            let mask = !((3 ^ code as u64) << sh);
            let old = w[i >> 5].fetch_and(mask, Ordering::Relaxed);
            old >> sh & 3 == 3
        };
        let id: [u8; 16] = std::array::from_fn(|i| i as u8);
        let mut count = 0usize;
        for ud in [p2.rank70(0x0f), p2.rank70(0xf0)] {
            for fb in [p2.rank70(0x0f), p2.rank70(0xf0)] {
                if set(self.canonical(ud as usize * 70 + fb as usize, &id, 0), 0) {
                    count += 1;
                }
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
                                    let (pair, tau, pll) = self.state(i);
                                    for k in 0..17 {
                                        let j = self.neighbour(p2, pair, &tau, pll, k);
                                        if get(j) == 3 && set(j, next_code) {
                                            local += 1;
                                        }
                                    }
                                } else if v == 3 {
                                    let (pair, tau, pll) = self.state(i);
                                    // All neighbours' entries prefetched, then checked.
                                    let mut js = [0usize; 17];
                                    for (k, j) in js.iter_mut().enumerate() {
                                        *j = self.neighbour(p2, pair, &tau, pll, k);
                                        crate::table::prefetch(&w[*j >> 5]);
                                    }
                                    if js.iter().any(|&j| get(j) == code) && set(i, next_code) {
                                        local += 1;
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
                    "  big phase-2 table depth {d}: {frontier} ({count} of {n}, {:.1} s)",
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // Every non-canonical entry gets its canonical entry's value.
        crate::par::for_chunks(n, 1 << 14, |r| {
            for i in r {
                let (pair, tau, pll) = self.state(i);
                let j = self.canonical(pair, &tau, pll);
                if j != i {
                    let c = get(j);
                    let sh = (i & 31) * 2;
                    let cur = w[i >> 5].load(Ordering::Relaxed) >> sh & 3;
                    if cur != c as u64 {
                        w[i >> 5].fetch_and(!(3 << sh), Ordering::Relaxed);
                        w[i >> 5].fetch_or((c as u64) << sh, Ordering::Relaxed);
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::MOVE_PERMS;
    use crate::phase2::P2_MOVES;

    #[test]
    fn classes_and_symmetry() {
        let p2 = P2Tables::build_centres_for_tests();
        let b = Big::new(&p2);
        eprintln!(
            "{} syms, {} classes, {} entries, {:.1} GB",
            b.syms.len(),
            b.classes(),
            b.n,
            b.bytes() as f64 / 1e9
        );
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..3000 {
            let idx = rng.gen_range(0..b.n);
            let (pair, tau, pll) = b.state(idx);
            assert_eq!(crate::comb::perm_parity(&tau[..12]), 0);
            let c = b.canonical(pair, &tau, pll);
            // Canonical indices decode to their own orbit.
            let (p2b, t2, l2) = b.state(c);
            assert_eq!(b.canonical(p2b, &t2, l2), c);
            // Symmetric images have the same canonical index.
            for s in &b.syms {
                assert_eq!(
                    b.canonical(s.pair[pair] as usize, &conj(&tau, &s.sym), pll),
                    c
                );
            }
            // The fast index is one of the stabiliser variants.
            let cc = (pair * 12 * 2) as u32 + pll;
            let (p3, t3, l3) = b.state(b.index(cc, &tau));
            assert_eq!(b.canonical(p3, &t3, l3), c);
        }
    }

    #[test]
    fn moves_match_cube() {
        // The pattern-pair moves agree with the cube.
        let p2 = P2Tables::build_centres_for_tests();
        let b = Big::new(&p2);
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut c = crate::cube::Cube::SOLVED;
        for _ in 0..2000 {
            let i = rng.gen_range(0..17);
            let cc = p2.centre_coord_of(&c);
            let (ud, fb) = ((cc / 1680) as usize, (cc / 24 % 70) as usize);
            c = c.apply_perm(&MOVE_PERMS[P2_MOVES[i] as usize]);
            let cc2 = p2.centre_coord_of(&c);
            assert_eq!(b.ud_move[ud][i] as u32, cc2 / 1680);
            assert_eq!(b.fb_move[fb][i] as u32, cc2 / 24 % 70);
        }
    }

    #[test]
    fn symmetry_reduced_bfs_matches_raw() {
        let p2 = P2Tables::build_centres_for_tests();
        let b = Big::new(&p2);
        let id: [u8; 16] = std::array::from_fn(|i| i as u8);
        let mut goals = vec![];
        for ud in [p2.rank70(0x0f), p2.rank70(0xf0)] {
            for fb in [p2.rank70(0x0f), p2.rank70(0xf0)] {
                goals.push((ud as usize * 70 + fb as usize, id, 0u32));
            }
        }
        let step = |&(pair, tau, pll): &(usize, [u8; 16], u32), k: usize| {
            let (ud, fb) = (pair / 70, pair % 70);
            (
                b.ud_move[ud][k] as usize * 70 + b.fb_move[fb][k] as usize,
                p2.tau_move(&tau, k),
                pll ^ (WIDE >> k & 1),
            )
        };
        let mut raw = std::collections::HashMap::new();
        let mut fr = goals.clone();
        for g in &goals {
            raw.insert(*g, 0u8);
        }
        for d in 1..=4u8 {
            let mut next = vec![];
            for s in &fr {
                for k in 0..17 {
                    let x = step(s, k);
                    if let std::collections::hash_map::Entry::Vacant(e) = raw.entry(x) {
                        e.insert(d);
                        next.push(x);
                    }
                }
            }
            fr = next;
        }
        let mut red = std::collections::HashMap::new();
        let mut fr: Vec<usize> = goals.iter().map(|g| b.canonical(g.0, &g.1, g.2)).collect();
        for &i in &fr {
            red.insert(i, 0u8);
        }
        for d in 1..=4u8 {
            let mut next = vec![];
            for &i in &fr {
                let s = b.state(i);
                for k in 0..17 {
                    let x = step(&s, k);
                    let c = b.canonical(x.0, &x.1, x.2);
                    if let std::collections::hash_map::Entry::Vacant(e) = red.entry(c) {
                        e.insert(d);
                        next.push(c);
                    }
                }
            }
            fr = next;
        }
        let mut bad = 0;
        for (s, &d) in &raw {
            if red.get(&b.canonical(s.0, &s.1, s.2)) != Some(&d) {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "raw states whose canonical entry has another depth");
    }
}
