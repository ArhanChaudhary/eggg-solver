//! A joint phase-2 pruning table over the pairing permutation tau, the U/D centre pattern and
//! the PLL-parity bit.
//!
//! The U/D and F/B quarter turns rotate the U/D centres and conjugate tau as set-up moves, and
//! the wide half turns both pair wings and swap centres between faces, so the pattern and the
//! pairing interact strongly; their joint distance is far tighter than the maximum of the two.
//!
//! Size: 70 * 12!/2 * 2 states, reduced by the four rotations V = {I, x2, y2, z2} (which keep
//! the low wing slots low and the U/D axis) and the swap of the U and D colours: the pair
//! (U/D pattern, tau on the four M-slice edges) is reduced to a class, and the other eight
//! entries of tau, conjugated into the class representative, are ranked (8!/2). Distances
//! are stored mod 3 in 2 bits (1.05 GB).

use crate::comb::{FACT, perm_rank};
use crate::cube::{EDGE_POS, GEO, ROTATIONS, centre_slot, mat_apply};
use crate::moves::MOVE_PERMS;
use crate::phase2::{P2_MOVES, P2Tables};
use crate::table::HugeBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// The M-slice edges UF, UB, DF, DB (Kociemba numbering) and the other eight.
const M_EDGES: [usize; 4] = [1, 3, 5, 7];
const REST: [usize; 8] = [0, 2, 4, 6, 8, 9, 10, 11];
const N_PREFIX: usize = 12 * 11 * 10 * 9;
const N_RAW: usize = 70 * N_PREFIX;
const N_REST: usize = 20160;
/// The wide half turns among the phase-2 moves (they flip the PLL bit).
const WIDE: u32 = 0b111 << 14;

struct Sym {
    /// Edge permutation (edge e goes to edge `e[e]`).
    e: [u8; 12],
    /// Image of a U/D pattern rank.
    ud: [u8; 70],
}

pub struct Joint {
    syms: Vec<Sym>,
    /// Raw (pattern, prefix) -> class << 2 | symmetry taking it to the class representative.
    cls: Vec<u32>,
    /// Class -> raw representative.
    rep: Vec<u32>,
    /// Class -> stabiliser bitmask (bit s: symmetry s fixes the representative), identity
    /// excluded. Symmetry 2k + c is rotation k of V, with the colour complement if c = 1.
    stab: Vec<u8>,
    pub ud_move: [[u8; 17]; 70],
    pub table: HugeBuf<u64>,
    pub n: usize,
}

#[inline(always)]
fn prefix_rank(tau: &[u8; 16]) -> usize {
    let mut used = 0u32;
    let mut r = 0usize;
    for (i, &e) in M_EDGES.iter().enumerate() {
        let v = tau[e] as u32;
        r = r * (12 - i) + (!used & ((1 << v) - 1)).count_ones() as usize;
        used |= 1 << v;
    }
    r
}

fn prefix_unrank(mut r: usize) -> [u8; 4] {
    let mut digits = [0usize; 4];
    for i in (0..4).rev() {
        digits[i] = r % (12 - i);
        r /= 12 - i;
    }
    let mut avail: Vec<u8> = (0..12).collect();
    let mut out = [0u8; 4];
    for i in 0..4 {
        out[i] = avail.remove(digits[i]);
    }
    out
}

/// Rank of the eight non-M entries of tau (relative to the values the prefix leaves), /2 for
/// the fixed parity.
#[inline(always)]
fn rest_rank(tau: &[u8; 16]) -> usize {
    let mut pm = 0u32;
    for &e in &M_EDGES {
        pm |= 1 << tau[e];
    }
    let mut p = [0u8; 8];
    for (i, &e) in REST.iter().enumerate() {
        let v = tau[e] as u32;
        p[i] = (v - (pm & ((1 << v) - 1)).count_ones()) as u8;
    }
    (perm_rank(&p) / 2) as usize
}

fn perm8_unrank(mut r: usize) -> [u8; 8] {
    let mut avail = [0u8, 1, 2, 3, 4, 5, 6, 7];
    let mut n = 8;
    let mut out = [0u8; 8];
    for i in 0..8 {
        let f = FACT[7 - i] as usize;
        let q = r / f;
        r %= f;
        out[i] = avail[q];
        avail.copy_within(q + 1..n, q);
        n -= 1;
    }
    out
}

fn conj(tau: &[u8; 16], s: &Sym) -> [u8; 16] {
    let mut t = *tau;
    for e in 0..12 {
        t[s.e[e] as usize] = s.e[tau[e] as usize];
    }
    t
}

impl Joint {
    /// Symmetries, classes and move tables (no table data).
    pub fn new(p2: &P2Tables) -> Joint {
        let g = &*GEO;
        let mut syms = vec![];
        // The four rotations of V, identity first.
        let mut mats: Vec<_> = ROTATIONS
            .iter()
            .filter(|m| (0..3).all(|i| (0..3).all(|j| (i == j) == (m[i][j] != 0))))
            .copied()
            .collect();
        mats.sort_by_key(|m| (m[0][0] != 1, m[1][1] != 1, m[2][2] != 1));
        assert_eq!(mats.len(), 4);
        for m in &mats {
            let mut e = [0u8; 12];
            for i in 0..12 {
                let p = mat_apply(m, EDGE_POS[i]);
                e[i] = EDGE_POS.iter().position(|&q| q == p).unwrap() as u8;
                // The low slot of edge i must go to the low slot of its image.
                let low = p2.low[i] as usize;
                let img = crate::cube::wing_slot(mat_apply(m, g.wing_pos[low]));
                assert_eq!(
                    img as u8, p2.low[e[i] as usize],
                    "symmetry does not keep low slots low"
                );
            }
            // U/D pattern: positions move, and the colours swap when U goes to D.
            let ud_pos = [0usize, 1, 2, 3, 12, 13, 14, 15];
            let comp = mat_apply(m, [0, 1, 0])[1] < 0;
            let mut ud = [0u8; 70];
            for r in 0..70 {
                let mask = p2.unrank70(r as u8);
                let mut out = 0u8;
                for (i, &p) in ud_pos.iter().enumerate() {
                    let q = centre_slot(mat_apply(m, g.centre_pos[p]));
                    let j = ud_pos.iter().position(|&x| x == q).unwrap();
                    if (mask >> i & 1 == 1) != comp {
                        out |= 1 << j;
                    }
                }
                ud[r] = p2.rank70(out);
            }
            // The distance is also invariant under swapping the U and D colours (a recolouring:
            // the goal has both orientations), and the moves outside the 17 (Lw2 = Rw2 x2...)
            // map neighbours of a symmetric image to complemented images: the symmetry group
            // must include the complement.
            let udc: [u8; 70] = std::array::from_fn(|r| p2.rank70(!p2.unrank70(ud[r])));
            syms.push(Sym { e, ud });
            syms.push(Sym { e, ud: udc });
        }
        // Classes of (pattern, prefix).
        let mut cls = vec![u32::MAX; N_RAW];
        let mut rep = vec![];
        let mut stab = vec![];
        for raw in 0..N_RAW {
            if cls[raw] != u32::MAX {
                continue;
            }
            let class = rep.len() as u32;
            rep.push(raw as u32);
            let (ud, pr) = (raw / N_PREFIX, raw % N_PREFIX);
            let pv = prefix_unrank(pr);
            let mut st = 0u8;
            for (k, s) in syms.iter().enumerate() {
                // Image under s: pattern s.ud[ud], prefix entries tau'[s(e)] = s(tau[e]).
                let mut t = [0u8; 16];
                for (i, &e) in M_EDGES.iter().enumerate() {
                    t[s.e[e] as usize] = s.e[pv[i] as usize];
                }
                let img = s.ud[ud] as usize * N_PREFIX + prefix_rank(&t);
                if img == raw && k != 0 {
                    st |= 1 << k;
                }
                if cls[img] == u32::MAX {
                    // The group is abelian and every element is its own inverse.
                    cls[img] = class << 3 | k as u32;
                }
            }
            stab.push(st);
        }
        let mut ud_move = [[0u8; 17]; 70];
        for r in 0..70u32 {
            let c = p2.centre_cube(r * 1680);
            for (i, &m) in P2_MOVES.iter().enumerate() {
                let cc = p2.centre_coord_of(&c.apply_perm(&MOVE_PERMS[m as usize]));
                ud_move[r as usize][i] = (cc / 1680) as u8;
            }
        }
        let n = rep.len() * N_REST * 2;
        Joint {
            syms,
            cls,
            rep,
            stab,
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

    /// Table index (one stabiliser variant; the table holds the same value in all of them).
    #[inline(always)]
    pub fn index(&self, ud: u32, tau: &[u8; 16], pll: u32) -> usize {
        let c = self.cls[ud as usize * N_PREFIX + prefix_rank(tau)];
        let s = (c & 7) as usize;
        let r = if s == 0 {
            rest_rank(tau)
        } else {
            rest_rank(&conj(tau, &self.syms[s]))
        };
        ((c >> 3) as usize * N_REST + r) * 2 + pll as usize
    }

    /// [`Joint::canonical`], for checks.
    pub fn canonical_pub(&self, ud: u32, tau: &[u8; 16], pll: u32) -> usize {
        self.canonical(ud, tau, pll)
    }

    /// The canonical index (minimum over the stabiliser), used by the generator.
    fn canonical(&self, ud: u32, tau: &[u8; 16], pll: u32) -> usize {
        let c = self.cls[ud as usize * N_PREFIX + prefix_rank(tau)];
        let class = (c >> 3) as usize;
        let s = (c & 7) as usize;
        let t = if s == 0 {
            *tau
        } else {
            conj(tau, &self.syms[s])
        };
        let mut r = rest_rank(&t);
        let st = self.stab[class];
        if st != 0 {
            for k in 1..8 {
                if st >> k & 1 == 1 {
                    r = r.min(rest_rank(&conj(&t, &self.syms[k])));
                }
            }
        }
        (class * N_REST + r) * 2 + pll as usize
    }

    /// A state with the given index.
    fn state(&self, idx: usize) -> (u32, [u8; 16], u32) {
        let pll = (idx & 1) as u32;
        let r = idx / 2 % N_REST;
        let class = idx / 2 / N_REST;
        let raw = self.rep[class] as usize;
        let (ud, pr) = (raw / N_PREFIX, raw % N_PREFIX);
        let pv = prefix_unrank(pr);
        let mut tau = [0u8; 16];
        for i in 12..16 {
            tau[i] = i as u8;
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
        let mut p = perm8_unrank(r * 2);
        let mut full = tau;
        for (i, &e) in REST.iter().enumerate() {
            full[e] = avail[p[i] as usize];
        }
        if crate::comb::perm_parity(&full[..12]) == 1 {
            p.swap(6, 7);
            for (i, &e) in REST.iter().enumerate() {
                full[e] = avail[p[i] as usize];
            }
        }
        (ud as u32, full, pll)
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> u8 {
        (self.table[i >> 5] >> ((i & 31) * 2) & 3) as u8
    }

    #[inline(always)]
    pub fn prefetch(&self, i: usize) {
        crate::table::prefetch(unsafe { self.table.as_ptr().add(i >> 5) });
    }

    /// Exact distance of a state, by walking down the mod-3 table.
    pub fn distance(&self, p2: &P2Tables, ud: u32, tau: &[u8; 16], pll: u32) -> u8 {
        let (mut ud, mut tau, mut pll) = (ud, *tau, pll);
        let mut d = 0;
        loop {
            let v = self.get(self.index(ud, &tau, pll));
            let mut next = None;
            for i in 0..17 {
                let t2 = p2.tau_move(&tau, i);
                let u2 = self.ud_move[ud as usize][i] as u32;
                let p = pll ^ (WIDE >> i & 1);
                if self.get(self.index(u2, &t2, p)) == (v + 2) % 3 {
                    next = Some((u2, t2, p));
                    break;
                }
            }
            match next {
                Some((u2, t2, p)) if d < 40 => {
                    ud = u2;
                    tau = t2;
                    pll = p;
                    d += 1;
                }
                _ => return d,
            }
        }
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
        let get = |i: usize| (w[i >> 5].load(Ordering::Relaxed) >> ((i & 31) * 2) & 3) as u8;
        // Set an unreached entry (code 3) to `code`; true if this call did it.
        let set = |i: usize, code: u8| -> bool {
            let sh = (i & 31) * 2;
            let mask = !((3 ^ code as u64) << sh);
            let old = w[i >> 5].fetch_and(mask, Ordering::Relaxed);
            old >> sh & 3 == 3
        };
        let id: [u8; 16] = std::array::from_fn(|i| i as u8);
        let mut count = 0usize;
        for ud in [p2.rank70(0x0f), p2.rank70(0xf0)] {
            if set(self.canonical(ud as u32, &id, 0), 0) {
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
                                    let (ud, tau, pll) = self.state(i);
                                    for k in 0..17 {
                                        let j = self.canonical(
                                            self.ud_move[ud as usize][k] as u32,
                                            &p2.tau_move(&tau, k),
                                            pll ^ (WIDE >> k & 1),
                                        );
                                        if get(j) == 3 && set(j, next_code) {
                                            local += 1;
                                        }
                                    }
                                } else if v == 3 {
                                    let (ud, tau, pll) = self.state(i);
                                    for k in 0..17 {
                                        let j = self.canonical(
                                            self.ud_move[ud as usize][k] as u32,
                                            &p2.tau_move(&tau, k),
                                            pll ^ (WIDE >> k & 1),
                                        );
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
                    "  joint depth {d}: {frontier} ({count} of {n}, {:.1} s)",
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        // Every non-canonical entry gets its canonical entry's value (backward passes visit
        // them with unreliable results).
        crate::par::for_chunks(n, 1 << 14, |r| {
            for i in r {
                let (ud, tau, pll) = self.state(i);
                let j = self.canonical(ud, &tau, pll);
                if j != i {
                    let c = get(j);
                    let sh = (i & 31) * 2;
                    let cur = w[i >> 5].load(Ordering::Relaxed) >> sh & 3;
                    if cur != c as u64 {
                        // Set bits to c: clear to 0 then OR c (entries of one word may be
                        // written by other threads, so use atomic ops).
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

    #[test]
    fn canonical_decodes_to_its_orbit() {
        let p2 = P2Tables::base_for_tests();
        let j = Joint::new(&p2);
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut bad = 0;
        for _ in 0..20000 {
            let idx = rng.gen_range(0..j.n);
            let (ud, tau, pll) = j.state(idx);
            let c = j.canonical(ud, &tau, pll);
            let (u2, t2, p2b) = j.state(c);
            if j.canonical(u2, &t2, p2b) != c {
                bad += 1;
            }
        }
        assert_eq!(
            bad, 0,
            "canonical indices that do not decode to their own orbit"
        );
    }

    #[test]
    fn symmetry_reduced_bfs_matches_raw() {
        let p2 = P2Tables::base_for_tests();
        let j = Joint::new(&p2);
        let id: [u8; 16] = std::array::from_fn(|i| i as u8);
        let goals = [
            (p2.rank70(0x0f) as u32, id, 0u32),
            (p2.rank70(0xf0) as u32, id, 0),
        ];
        // Raw BFS.
        let mut raw = std::collections::HashMap::new();
        let mut fr: Vec<(u32, [u8; 16], u32)> = goals.to_vec();
        for g in &goals {
            raw.insert(*g, 0u8);
        }
        for d in 1..=4u8 {
            let mut next = vec![];
            for &(ud, tau, pll) in &fr {
                for k in 0..17 {
                    let s = (
                        j.ud_move[ud as usize][k] as u32,
                        p2.tau_move(&tau, k),
                        pll ^ (WIDE >> k & 1),
                    );
                    if !raw.contains_key(&s) {
                        raw.insert(s, d);
                        next.push(s);
                    }
                }
            }
            fr = next;
        }
        // Symmetry-reduced BFS over canonical indices.
        let mut red = std::collections::HashMap::new();
        let mut fr: Vec<usize> = goals.iter().map(|g| j.canonical(g.0, &g.1, g.2)).collect();
        for &i in &fr {
            red.insert(i, 0u8);
        }
        for d in 1..=4u8 {
            let mut next = vec![];
            for &i in &fr {
                let (ud, tau, pll) = j.state(i);
                for k in 0..17 {
                    let c = j.canonical(
                        j.ud_move[ud as usize][k] as u32,
                        &p2.tau_move(&tau, k),
                        pll ^ (WIDE >> k & 1),
                    );
                    if !red.contains_key(&c) {
                        red.insert(c, d);
                        next.push(c);
                    }
                }
            }
            fr = next;
        }
        // Which symmetries change raw depths (for states deep enough inside the BFS ball)?
        let mut per_sym = [0usize; 4];
        let mut per_sym_nocomp = [0usize; 4];
        for (s, &d) in &raw {
            if d > 2 {
                continue;
            }
            for (k, sy) in j.syms.iter().enumerate() {
                let img = (sy.ud[s.0 as usize] as u32, conj(&s.1, sy), s.2);
                if raw.get(&img) != Some(&d) {
                    per_sym[k] += 1;
                }
                // Same without the colour complement.
                let comp = sy.ud[p2.rank70(0x0f) as usize] != p2.rank70(0x0f);
                let ud2 = if comp {
                    p2.rank70(!p2.unrank70(sy.ud[s.0 as usize]) as u8) as u32
                } else {
                    sy.ud[s.0 as usize] as u32
                };
                if raw.get(&(ud2, conj(&s.1, sy), s.2)) != Some(&d) {
                    per_sym_nocomp[k] += 1;
                }
            }
        }
        eprintln!(
            "depth changes per symmetry: {per_sym:?}, without complement: {per_sym_nocomp:?}"
        );
        let mut bad = 0;
        for (s, &d) in &raw {
            let c = j.canonical(s.0, &s.1, s.2);
            if red.get(&c) != Some(&d) {
                bad += 1;
                if bad <= 4 {
                    let rep = j.state(c);
                    eprintln!(
                        "raw depth {d}, reduced {:?}; rep raw depth {:?}; rep canonical == c: {}; ud {} rep ud {}",
                        red.get(&c),
                        raw.get(&rep),
                        j.canonical(rep.0, &rep.1, rep.2) == c,
                        s.0,
                        rep.0
                    );
                }
            }
        }
        assert_eq!(bad, 0, "raw states whose canonical entry has another depth");
    }

    #[test]
    fn raw_level_counts() {
        // Raw BFS over (U/D pattern, tau, PLL bit) from the goals; the counts were measured
        // independently with a separate model of the phase-2 group.
        let p2 = P2Tables::base_for_tests();
        let j = Joint::new(&p2);
        let id: [u8; 16] = std::array::from_fn(|i| i as u8);
        let mut seen = std::collections::HashSet::new();
        let mut frontier: Vec<(u32, [u8; 16], u32)> = vec![];
        for ud in [p2.rank70(0x0f), p2.rank70(0xf0)] {
            frontier.push((ud as u32, id, 0));
            seen.insert((ud as u32, id, 0));
        }
        let mut counts = vec![frontier.len()];
        for _ in 0..6 {
            let mut next = vec![];
            for &(ud, tau, pll) in &frontier {
                for k in 0..17 {
                    let s = (
                        j.ud_move[ud as usize][k] as u32,
                        p2.tau_move(&tau, k),
                        pll ^ (WIDE >> k & 1),
                    );
                    if seen.insert(s) {
                        next.push(s);
                    }
                }
            }
            counts.push(next.len());
            frontier = next;
        }
        assert_eq!(counts, vec![2, 6, 48, 418, 3708, 30460, 260796]);
    }

    #[test]
    fn index_roundtrip_and_symmetry() {
        let p2 = P2Tables::base_for_tests();
        let j = Joint::new(&p2);
        assert!(
            j.classes() > 100_000 && j.classes() < 110_000,
            "{} classes",
            j.classes()
        );
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..2000 {
            let idx = rng.gen_range(0..j.n);
            let (ud, tau, pll) = j.state(idx);
            assert_eq!(crate::comb::perm_parity(&tau[..12]), 0);
            let c = j.canonical(ud, &tau, pll);
            // Symmetric images have the same canonical index.
            for s in &j.syms {
                assert_eq!(
                    j.canonical(s.ud[ud as usize] as u32, &conj(&tau, s), pll),
                    c
                );
            }
            // The fast index is one of the stabiliser variants.
            let (u2, t2, p2b) = j.state(j.index(ud, &tau, pll));
            assert_eq!(j.canonical(u2, &t2, p2b), c);
        }
    }
}
