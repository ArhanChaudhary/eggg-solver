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

use crate::comb::FACT;
use crate::cube::{EDGE_POS, GEO, ROTATIONS, centre_slot, mat_apply};
use crate::moves::MOVE_PERMS;
use crate::phase2::{P2_MOVES, P2Tables};
use crate::simd::V32;
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
    /// [`conj`] as two byte shuffles (low lane): gather the edges by the inverse position
    /// permutation, then relabel them.
    e_inv: V32,
    e_map: V32,
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
    /// [`P2Tables::tau_move`] as two byte shuffles per move: (pl_inv, ph).
    tau_moves: [(V32, V32); 17],
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
/// the fixed parity: their Lehmer code over the values not used by the prefix.
#[inline(always)]
fn rest_rank(tau: &[u8; 16]) -> usize {
    let mut avail = !0u32;
    for &e in &M_EDGES {
        avail &= !(1 << tau[e]);
    }
    let mut r = 0u64;
    for (i, &e) in REST.iter().enumerate() {
        let v = tau[e] as u32;
        r += (avail & ((1 << v) - 1)).count_ones() as u64 * FACT[7 - i];
        avail &= !(1 << v);
    }
    (r / 2) as usize
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

/// The edge part of [`conj`] (entries 12.. undefined), with byte shuffles.
#[inline(always)]
fn conj_edges(tau: &[u8; 16], s: &Sym) -> [u8; 16] {
    let mut v = V32::ZERO;
    v.0[..16].copy_from_slice(tau);
    let t = crate::simd::shuf(&s.e_map, &crate::simd::shuf(&v, &s.e_inv));
    t.0[..16].try_into().unwrap()
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
            let (mut e_inv, mut e_map) = (V32([0x80; 32]), V32([0; 32]));
            for i in 0..12 {
                e_inv.0[e[i] as usize] = i as u8;
                e_map.0[i] = e[i];
            }
            syms.push(Sym {
                e,
                ud,
                e_inv,
                e_map,
            });
            syms.push(Sym {
                e,
                ud: udc,
                e_inv,
                e_map,
            });
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
        let tau_moves = std::array::from_fn(|i| {
            let (mut pl, mut ph) = (V32::ZERO, V32::ZERO);
            pl.0[..16].copy_from_slice(&p2.pl_inv[i]);
            ph.0[..16].copy_from_slice(&p2.ph[i]);
            (pl, ph)
        });
        Joint {
            syms,
            cls,
            rep,
            stab,
            ud_move,
            tau_moves,
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
            rest_rank(&conj_edges(tau, &self.syms[s]))
        };
        ((c >> 3) as usize * N_REST + r) * 2 + pll as usize
    }

    /// [`Joint::index`] of a tau held in the low half of a vector.
    #[inline(always)]
    fn index_v(&self, ud: u32, tau: &V32, pll: u32) -> usize {
        let t: &[u8; 16] = tau.0[..16].try_into().unwrap();
        let c = self.cls[ud as usize * N_PREFIX + prefix_rank(t)];
        let s = (c & 7) as usize;
        let r = if s == 0 {
            rest_rank(t)
        } else {
            let sy = &self.syms[s];
            let u = crate::simd::shuf(&sy.e_map, &crate::simd::shuf(tau, &sy.e_inv));
            rest_rank(u.0[..16].try_into().unwrap())
        };
        ((c >> 3) as usize * N_REST + r) * 2 + pll as usize
    }

    /// [`P2Tables::tau_move`] of a tau held in the low half of a vector.
    #[inline(always)]
    fn tau_move_v(&self, tau: &V32, i: usize) -> V32 {
        let (pl, ph) = &self.tau_moves[i];
        crate::simd::shuf(ph, &crate::simd::shuf(tau, pl))
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

    /// Exact distance of a state, by walking down the mod-3 table (four children read at a
    /// time).
    pub fn distance(&self, ud: u32, tau: &[u8; 16], pll: u32) -> u8 {
        let mut w = Walk::new(self, ud, tau, pll, u8::MAX);
        loop {
            if let Some(d) = w.step::<4>(self) {
                return d.expect("no bound");
            }
        }
    }

    /// (steps, children read) of a walk from a state trying the children in `order` (with the
    /// restriction of [`AFTER`]), or all 17 in index order (`order` None).
    #[cfg(test)]
    pub(crate) fn walk_reads(
        &self,
        p2: &P2Tables,
        ud: u32,
        tau: &[u8; 16],
        pll: u32,
        order: Option<&[u8; 17]>,
    ) -> (usize, usize) {
        let ident: [u8; 17] = std::array::from_fn(|i| i as u8);
        let after = order.map(after_table).unwrap_or([(1 << 17) - 1; 18]);
        let order = order.unwrap_or(&ident);
        let (mut ud, mut tau, mut pll) = (ud, *tau, pll);
        let (mut steps, mut reads) = (0, 0);
        let mut last = 17;
        loop {
            let v = self.get(self.index(ud, &tau, pll));
            let mut next = None;
            for k in 0..17 {
                if after[last] >> k & 1 == 0 {
                    continue;
                }
                let i = order[k] as usize;
                let t2 = p2.tau_move(&tau, i);
                let u2 = self.ud_move[ud as usize][i] as u32;
                let p = pll ^ (WIDE >> i & 1);
                reads += 1;
                if self.get(self.index(u2, &t2, p)) == (v + 2) % 3 {
                    next = Some((u2, t2, p, i));
                    break;
                }
            }
            match next {
                Some((u2, t2, p, i)) => {
                    (ud, tau, pll, last) = (u2, t2, p, i);
                    steps += 1;
                }
                None => return (steps, reads),
            }
        }
    }

    /// The former [`Joint::distance`] (all 17 children tried at each step, one by one, and the
    /// former index computation).
    #[cfg(test)]
    pub(crate) fn distance_ref(&self, p2: &P2Tables, ud: u32, tau: &[u8; 16], pll: u32) -> u8 {
        let index = |ud: u32, tau: &[u8; 16], pll: u32| {
            let c = self.cls[ud as usize * N_PREFIX + prefix_rank(tau)];
            let s = (c & 7) as usize;
            let r = if s == 0 {
                tests::rest_rank_ref(tau)
            } else {
                tests::rest_rank_ref(&conj_edges(tau, &self.syms[s]))
            };
            ((c >> 3) as usize * N_REST + r) * 2 + pll as usize
        };
        let (mut ud, mut tau, mut pll) = (ud, *tau, pll);
        let mut d = 0;
        loop {
            let v = self.get(index(ud, &tau, pll));
            let mut next = None;
            for i in 0..17 {
                let t2 = p2.tau_move(&tau, i);
                let u2 = self.ud_move[ud as usize][i] as u32;
                let p = pll ^ (WIDE >> i & 1);
                if self.get(index(u2, &t2, p)) == (v + 2) % 3 {
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

    /// Exact distances of states (U/D pattern, tau, PLL bit, bound), or None for those whose
    /// distance exceeds their bound (found out before walking all the way down). The walks are
    /// interleaved, one child at a time each, so that each one's table read is in flight while
    /// the others compute (a walk is a chain of dependent reads, and computing a child's index
    /// costs about as much as a read).
    pub fn distances_at_most(
        &self,
        states: &[(u32, [u8; 16], u32, u8)],
        out: &mut Vec<Option<u8>>,
    ) {
        const LANES: usize = 16;
        out.clear();
        out.resize(states.len(), None);
        let mut lanes: Vec<(usize, Walk)> = Vec::with_capacity(LANES);
        let mut next = 0;
        loop {
            while lanes.len() < LANES && next < states.len() {
                let (ud, tau, pll, max) = states[next];
                lanes.push((next, Walk::new(self, ud, &tau, pll, max)));
                next += 1;
            }
            if lanes.is_empty() {
                return;
            }
            let mut k = 0;
            while k < lanes.len() {
                if let Some(d) = lanes[k].1.step::<1>(self) {
                    out[lanes[k].0] = d;
                    lanes.swap_remove(k);
                } else {
                    k += 1;
                }
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

/// The order in which a walk tries the children (phase-2 move indices): the wide half turns
/// (each one move closer at ~20% of the steps of a walk from a phase-1 endpoint, against 3-8%
/// for the others), the quarter turns, then the half turns. Walks from phase-1 endpoints then
/// read ~73 children instead of ~106 in index order (~146 without the restriction).
pub(crate) const ORDER: [u8; 17] = [15, 16, 14, 0, 2, 4, 6, 7, 9, 11, 13, 3, 10, 1, 5, 8, 12];

/// The children a walk tries after phase-2 move `last` (17: none), as bits of positions in
/// [`ORDER`]: not the same face as `last`, nor of its axis and before it in the order.
const AFTER: [u32; 18] = after_table(&ORDER);

const fn after_table(order: &[u8; 17]) -> [u32; 18] {
    let mut pos = [0usize; 17];
    let mut k = 0;
    while k < 17 {
        pos[order[k] as usize] = k;
        k += 1;
    }
    let mut t = [0u32; 18];
    let mut last = 0;
    while last <= 17 {
        let mut k = 0;
        while k < 17 {
            let m = order[k] as usize;
            let ok = last == 17 || {
                let (fl, fm) = (P2_MOVES[last] / 3, P2_MOVES[m] / 3);
                fl != fm && (fl % 3 != fm % 3 || k > pos[last])
            };
            if ok {
                t[last] |= 1 << k;
            }
            k += 1;
        }
        last += 1;
    }
    t
}

/// A walk down the joint table from a state to the goal, one child one move closer per step
/// (the first in [`ORDER`]). Such a child is never a move of the same face as the step that
/// led to the state, nor of its axis and before it in the order (the moves of an axis commute,
/// so that move would have been one move closer from the parent, and tried first): those are
/// skipped, and the walk takes the same path as one trying all 17 children.
struct Walk {
    ud: u32,
    /// tau in the low 16 bytes.
    tau: V32,
    pll: u32,
    /// The current state's stored value (its distance mod 3; 3 = never reached; `UNREAD`
    /// before the first step).
    v: u8,
    /// Steps taken.
    d: u8,
    /// The distance bound: the walk ends without a distance once it must exceed it.
    max: u8,
    /// Children of the current state not tried yet (move index bits).
    left: u32,
    /// Children whose entries are being fetched: (index, U/D pattern, tau, PLL bit, move).
    pend: [(usize, u32, V32, u32, u8); 4],
    npend: usize,
}

const UNREAD: u8 = 0xff;

impl Walk {
    /// A walk from a state, whose own entry is fetched (and read by the first step).
    fn new(j: &Joint, ud: u32, tau: &[u8; 16], pll: u32, max: u8) -> Walk {
        let mut t = V32::ZERO;
        t.0[..16].copy_from_slice(tau);
        let idx = j.index(ud, tau, pll);
        j.prefetch(idx);
        let mut pend = [(0, 0, V32::ZERO, 0, 0); 4];
        pend[0].0 = idx;
        Walk {
            ud,
            tau: t,
            pll,
            v: UNREAD,
            d: 0,
            max,
            left: AFTER[17],
            pend,
            npend: 0,
        }
    }

    /// Read the pending children (fetched by the last call), then fetch the next `G`: Some once
    /// the walk is over (Some(None): the distance exceeds the bound).
    #[inline(always)]
    fn step<const G: usize>(&mut self, j: &Joint) -> Option<Option<u8>> {
        if self.v == UNREAD {
            self.v = j.get(self.pend[0].0);
            return self.fetch::<G>(j);
        }
        let want = (self.v + 2) % 3;
        for k in 0..self.npend {
            let (idx, ud, tau, pll, i) = self.pend[k];
            if j.get(idx) == want {
                if self.d >= 40 {
                    return Some(Some(self.d));
                }
                (self.ud, self.tau, self.pll, self.v) = (ud, tau, pll, want);
                self.d += 1;
                self.left = AFTER[i as usize];
                self.npend = 0;
                return self.fetch::<G>(j);
            }
        }
        self.npend = 0;
        self.fetch::<G>(j)
    }

    #[inline(always)]
    fn fetch<const G: usize>(&mut self, j: &Joint) -> Option<Option<u8>> {
        // The rest of the walk takes at least v moves (its length is v mod 3).
        if self.v < 3 && self.d + self.v > self.max {
            return Some(None);
        }
        if self.left == 0 {
            return Some(Some(self.d));
        }
        while self.left != 0 && self.npend < G {
            let i = ORDER[self.left.trailing_zeros() as usize] as usize;
            self.left &= self.left - 1;
            let tau = j.tau_move_v(&self.tau, i);
            let ud = j.ud_move[self.ud as usize][i] as u32;
            let pll = self.pll ^ (WIDE >> i & 1);
            let idx = j.index_v(ud, &tau, pll);
            j.prefetch(idx);
            self.pend[self.npend] = (idx, ud, tau, pll, i as u8);
            self.npend += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_order_restriction() {
        // Each move's own face is excluded, other axes are allowed, and of two moves of one
        // axis (different faces) exactly one may follow the other.
        for a in 0..17 {
            assert_eq!(ORDER.iter().filter(|&&m| m as usize == a).count(), 1);
            for k in 0..17 {
                let b = ORDER[k] as usize;
                let (fa, fb) = (P2_MOVES[a] / 3, P2_MOVES[b] / 3);
                let ab = AFTER[a] >> k & 1 == 1;
                let kb = ORDER.iter().position(|&m| m as usize == a).unwrap();
                let ba = AFTER[b] >> kb & 1 == 1;
                if fa == fb {
                    assert!(!ab && !ba);
                } else if fa % 3 != fb % 3 {
                    assert!(ab && ba);
                } else {
                    assert!(ab != ba);
                }
            }
        }
        assert_eq!(AFTER[17], (1 << 17) - 1);
    }

    /// The former [`rest_rank`]: relabel the eight values to 0..8, then rank.
    pub(super) fn rest_rank_ref(tau: &[u8; 16]) -> usize {
        let mut pm = 0u32;
        for &e in &M_EDGES {
            pm |= 1 << tau[e];
        }
        let mut p = [0u8; 8];
        for (i, &e) in REST.iter().enumerate() {
            let v = tau[e] as u32;
            p[i] = (v - (pm & ((1 << v) - 1)).count_ones()) as u8;
        }
        (crate::comb::perm_rank(&p) / 2) as usize
    }

    #[test]
    fn rest_rank_matches_relabelled_lehmer_rank() {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        for _ in 0..20000 {
            let mut tau: [u8; 16] = std::array::from_fn(|i| i as u8);
            tau[..12].shuffle(&mut rng);
            assert_eq!(rest_rank(&tau), rest_rank_ref(&tau));
        }
    }

    #[test]
    fn conj_edges_matches_conj() {
        let p2 = P2Tables::base_for_tests();
        let j = Joint::new(&p2);
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..2000 {
            let (_, tau, _) = j.state(rng.gen_range(0..j.n));
            for s in &j.syms {
                assert_eq!(conj_edges(&tau, s)[..12], conj(&tau, s)[..12]);
            }
        }
    }

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
                    if let std::collections::hash_map::Entry::Vacant(e) = raw.entry(s) {
                        e.insert(d);
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
