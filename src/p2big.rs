//! Large joint phase-2 pruning tables over the pairing permutation tau, two of the three
//! centre-pattern coordinates and the PLL-parity bit:
//!
//! * [`Kind::UdRl`]: U/D pattern (70) x R/L pattern (12) x tau x PLL, read twice: directly and
//!   through the rotation x, which turns the F/B pattern into a U/D one (as the F/B view of the
//!   1 GB joint table does). 4.0e11 states, 16 symmetries, ~6.3 GB.
//! * [`Kind::UdFb`]: U/D pattern x F/B pattern x tau x PLL (everything of the phase-2 state but
//!   the R/L pattern). 2.35e12 states, 32 symmetries, ~18.4 GB.
//!
//! Each dominates the 1 GB joint table (tau x U/D x PLL). Measured with their exact values
//! computed by IDA* (examples/p2deep, ORACLE=...), phase-2 nodes per solution when
//! enumerating all solutions one move above the bound, relative to the two views of the 1 GB
//! table: U/D x R/L (both views) 3.8x fewer, U/D x F/B 2.4x fewer, all three 6.6x fewer.
//!
//! Symmetries: the spatial symmetries (rotations and reflections) that keep the low wing
//! slots low and the axes of the two patterns (as a pair: U/D x F/B also has rotoreflections
//! of order 4 about the R/L axis that swap the U/D and F/B axes), times the colour swaps of
//! the two patterns (the projected goal has every orientation of the two solved patterns).
//! The pair of patterns is reduced first, then the M-slice entries of tau under the pair's
//! stabiliser; the other eight entries are ranked (8!/2). Distances are stored mod 3 in
//! 2 bits.

use crate::cube::{
    B, D, EDGE_POS, F, FACE_NORMAL, GEO, L, Mat, R, U, centre_slot, face_of_normal, mat_apply,
};
use crate::p2joint::{
    M_EDGES, N_PREFIX, N_REST, Sym, WIDE, conj, conj_edges, perm8_unrank, prefix_rank,
    prefix_unrank, rest_rank,
};
use crate::phase2::P2Tables;
use crate::simd::V32;
use crate::table::HugeBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Centre positions of the U/D, R/L and F/B orbits (face +, face -).
const POS: [[usize; 8]; 3] = [
    [0, 1, 2, 3, 12, 13, 14, 15],
    [4, 5, 6, 7, 16, 17, 18, 19],
    [8, 9, 10, 11, 20, 21, 22, 23],
];
/// The colour of each orbit marked in its pattern mask, and the other one.
const FIRST: [usize; 3] = [U, R, F];
const SECOND: [usize; 3] = [D, L, B];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// U/D x R/L patterns (with the x view: F/B x R/L).
    UdRl,
    /// U/D x F/B patterns.
    UdFb,
}

impl Kind {
    /// Orbit (0 U/D, 1 R/L, 2 F/B) of the second pattern.
    fn second(self) -> usize {
        match self {
            Kind::UdRl => 1,
            Kind::UdFb => 2,
        }
    }

    /// Number of values of the second pattern.
    fn nb(self) -> usize {
        match self {
            Kind::UdRl => 12,
            Kind::UdFb => 70,
        }
    }

    pub fn file_name(self) -> &'static str {
        match self {
            Kind::UdRl => "p2-big-udrl",
            Kind::UdFb => "p2-big-udfb",
        }
    }
}

struct BSym {
    /// Edge part (conjugation of tau).
    sym: Sym,
    /// The edge part is the identity (a pure colour swap).
    plain: bool,
    /// Image of a pattern pair.
    pair: Vec<u16>,
}

#[derive(Clone, Copy)]
struct PairEntry {
    /// Symmetry taking the pair to its class representative.
    sym: u8,
    /// The representative has a non-trivial stabiliser.
    nt: bool,
    /// Trivial stabiliser: the first class of the pair class (class = base + prefix rank,
    /// most pairs); otherwise the pair class's offset in `pref_cls`.
    base: u32,
}

/// The table read through the rotation x (U/D x R/L only): the state's (F/B, R/L) patterns
/// become the rotated state's (U/D, R/L) ones, and the rotated pairing is the conjugate of
/// tau's inverse (x takes the low wing slots to the high ones).
pub struct View {
    /// Rotated pattern pair, by fb * 12 + rl.
    pair: Vec<u16>,
    sym: Sym,
    inverse: bool,
}

pub struct Big {
    pub kind: Kind,
    syms: Vec<BSym>,
    /// Pattern pair -> the symmetry taking it to its class representative, and where its
    /// classes are.
    pair_cls: Vec<PairEntry>,
    /// (pair class with a non-trivial stabiliser, M-slice prefix) -> class << 6 | (the class
    /// has a non-trivial stabiliser) << 5 | symmetry (in the pair's stabiliser) taking the
    /// prefix to the class representative.
    pref_cls: Vec<u32>,
    /// Class -> representative (pair, prefix).
    rep: Vec<(u16, u16)>,
    /// Class -> stabiliser of the representative (bit s: symmetry s fixes it), identity
    /// excluded.
    stab: Vec<u32>,
    /// Pattern moves of the two orbits: [pattern][move].
    a_move: Vec<[u8; 17]>,
    b_move: Vec<[u8; 17]>,
    pub view: Option<View>,
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

/// Pattern masks of the three orbits (bit i: position `POS[o][i]` holds colour `FIRST[o]`)
/// after conjugating by `m`: a centre of colour c at position p goes to position m(p) with
/// colour m(c).
fn image_masks(m: &Mat, masks: [u8; 3]) -> [u8; 3] {
    let g = &*GEO;
    let col = |f: usize| face_of_normal(mat_apply(m, FACE_NORMAL[f]));
    let mut out = [0u8; 3];
    for o in 0..3 {
        for (i, &p) in POS[o].iter().enumerate() {
            let c = if masks[o] >> i & 1 == 1 {
                FIRST[o]
            } else {
                SECOND[o]
            };
            let q = centre_slot(mat_apply(m, g.centre_pos[p]));
            let (o2, j) = (0..3)
                .find_map(|o2| POS[o2].iter().position(|&x| x == q).map(|j| (o2, j)))
                .unwrap();
            let c2 = col(c);
            assert!(c2 == FIRST[o2] || c2 == SECOND[o2]);
            if c2 == FIRST[o2] {
                out[o2] |= 1 << j;
            }
        }
    }
    out
}

/// The edge map of `m` and whether it keeps the low wing slots low (None: it mixes them).
fn edge_map(p2: &P2Tables, m: &Mat) -> ([u8; 12], Option<bool>) {
    let g = &*GEO;
    let mut e = [0u8; 12];
    let mut lows = 0;
    for i in 0..12 {
        let p = mat_apply(m, EDGE_POS[i]);
        e[i] = EDGE_POS.iter().position(|&q| q == p).unwrap() as u8;
        let img = crate::cube::wing_slot(mat_apply(m, g.wing_pos[p2.low[i] as usize]));
        if img as u8 == p2.low[e[i] as usize] {
            lows += 1;
        }
    }
    let keep = match lows {
        12 => Some(true),
        0 => Some(false),
        _ => None,
    };
    (e, keep)
}

fn edge_sym(e: [u8; 12]) -> Sym {
    let (mut e_inv, mut e_map) = (V32([0x80; 32]), V32([0; 32]));
    for i in 0..12 {
        e_inv.0[e[i] as usize] = i as u8;
        e_map.0[i] = e[i];
    }
    Sym {
        e,
        ud: [0; 70],
        e_inv,
        e_map,
    }
}

impl Big {
    /// Rank of orbit `o`'s pattern mask (70 for U/D and F/B, 12 for R/L).
    fn rank(p2: &P2Tables, o: usize, mask: u8) -> usize {
        let r = if o == 1 {
            p2.rank12(mask)
        } else {
            p2.rank70(mask)
        };
        assert!(r != 255, "pattern not allowed");
        r as usize
    }

    fn unrank(p2: &P2Tables, o: usize, r: usize) -> u8 {
        if o == 1 {
            p2.unrank12(r as u8)
        } else {
            p2.unrank70(r as u8)
        }
    }

    /// Symmetries, classes and move tables (no table data).
    pub fn new(p2: &P2Tables, kind: Kind) -> Big {
        let ob = kind.second();
        let nb = kind.nb();
        let n_pair = 70 * nb;
        let mut geo: Vec<(Mat, [u8; 12])> = vec![];
        for m in all_mats() {
            // Keep the R/L axis, and for U/D x R/L also the U/D axis.
            if m[0][0] == 0 || (kind == Kind::UdRl && m[1][1] == 0) {
                continue;
            }
            let (e, keep) = edge_map(p2, &m);
            assert!(keep.is_some(), "a symmetry mixes low and high slots");
            if keep == Some(true) {
                geo.push((m, e));
            }
        }
        // Identity first.
        geo.sort_by_key(|(m, _)| *m != crate::cube::IDENTITY);
        assert_eq!(
            geo.len(),
            if kind == Kind::UdRl { 4 } else { 8 },
            "low-keeping symmetries"
        );
        let mut syms = vec![];
        for (m, e) in &geo {
            let base: Vec<(u8, u8)> = (0..n_pair)
                .map(|pair| {
                    let mut masks = [0x0fu8; 3];
                    masks[0] = Self::unrank(p2, 0, pair / nb);
                    masks[ob] = Self::unrank(p2, ob, pair % nb);
                    let out = image_masks(m, masks);
                    (out[0], out[ob])
                })
                .collect();
            let plain = *m == crate::cube::IDENTITY;
            for comp in 0..4u32 {
                let pair = base
                    .iter()
                    .map(|&(a, b)| {
                        let a = if comp & 1 == 1 { !a } else { a };
                        let b = if comp & 2 == 2 { !b } else { b };
                        (Self::rank(p2, 0, a) * nb + Self::rank(p2, ob, b)) as u16
                    })
                    .collect();
                syms.push(BSym {
                    sym: edge_sym(*e),
                    plain,
                    pair,
                });
            }
        }
        // Inverses (U/D x F/B's group is not abelian).
        let inv: Vec<u32> = syms
            .iter()
            .map(|s| {
                syms.iter()
                    .position(|t| {
                        (0..n_pair).all(|p| t.pair[s.pair[p] as usize] as usize == p)
                            && (0..12).all(|i| t.sym.e[s.sym.e[i] as usize] as usize == i)
                    })
                    .expect("symmetry without an inverse") as u32
            })
            .collect();
        let ns = syms.len();
        // Pair classes.
        let mut pair_cls = vec![u32::MAX; n_pair];
        let mut pair_rep = vec![];
        let mut pair_stab = vec![];
        for p in 0..n_pair {
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
        // Prefix classes within each pair class: one per prefix when the pair's stabiliser is
        // trivial, else the orbits of the stabiliser.
        let mut pref_cls = vec![];
        let mut rep = vec![];
        let mut stab = vec![];
        let mut info = vec![(0u32, false); pair_rep.len()];
        for (c1, &p) in pair_rep.iter().enumerate() {
            let st1 = pair_stab[c1];
            if st1 == 1 {
                info[c1] = (rep.len() as u32, false);
                for pr in 0..N_PREFIX {
                    rep.push((p as u16, pr as u16));
                    stab.push(0);
                }
                continue;
            }
            let off = pref_cls.len();
            info[c1] = (off as u32, true);
            pref_cls.resize(off + N_PREFIX, u32::MAX);
            for pr in 0..N_PREFIX {
                if pref_cls[off + pr] != u32::MAX {
                    continue;
                }
                let class = rep.len() as u32;
                rep.push((p as u16, pr as u16));
                let pv = prefix_unrank(pr);
                let mut st = 0u32;
                let mut imgs = vec![];
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
                    imgs.push((img, k));
                }
                for (img, k) in imgs {
                    let slot = &mut pref_cls[off + img];
                    if *slot == u32::MAX {
                        *slot = class << 6 | ((st != 0) as u32) << 5 | inv[k];
                    }
                }
                stab.push(st);
            }
        }
        let pair_cls = pair_cls
            .iter()
            .map(|&c| {
                let (base, nt) = info[(c >> 5) as usize];
                PairEntry {
                    sym: (c & 31) as u8,
                    nt,
                    base,
                }
            })
            .collect();
        assert!(ns <= 32 && rep.len() < 1 << 26);
        // Pattern moves, read off the centre coordinate's (((ud * 70 + fb) * 12 + rl) * 2 + pll).
        let unit = [1680usize, 2, 24];
        let modulo = [70usize, 12, 70];
        let moves = |o: usize| -> Vec<[u8; 17]> {
            (0..modulo[o])
                .map(|r| {
                    std::array::from_fn(|i| {
                        (p2.centre_move[r * unit[o]][i] as usize / unit[o] % modulo[o]) as u8
                    })
                })
                .collect()
        };
        let view = (kind == Kind::UdRl).then(|| {
            let m = *crate::cube::ROTATIONS
                .iter()
                .find(|m| m[0][0] == 1 && m[1][1] == 0)
                .unwrap();
            let (e, keep) = edge_map(p2, &m);
            let pair = (0..70 * 12)
                .map(|x| {
                    let masks = [
                        0x0f,
                        Self::unrank(p2, 1, x % 12),
                        Self::unrank(p2, 2, x / 12),
                    ];
                    let out = image_masks(&m, masks);
                    (Self::rank(p2, 0, out[0]) * 12 + Self::rank(p2, 1, out[1])) as u16
                })
                .collect();
            View {
                pair,
                sym: edge_sym(e),
                inverse: !keep.expect("x mixes low and high slots"),
            }
        });
        let n = rep.len() * N_REST * 2;
        Big {
            kind,
            syms,
            pair_cls,
            pref_cls,
            rep,
            stab,
            a_move: moves(0),
            b_move: moves(ob),
            view,
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

    /// The pattern pair of a full centre coordinate.
    #[inline(always)]
    fn pair_of(&self, cc: u32) -> usize {
        match self.kind {
            Kind::UdRl => (cc / 1680 * 12 + cc / 2 % 12) as usize,
            Kind::UdFb => (cc / 24) as usize,
        }
    }

    /// The class, the tau conjugated into its representative's frame (one stabiliser
    /// variant) and whether the class has a non-trivial stabiliser.
    #[inline(always)]
    fn class_of(&self, pair: usize, tau: &[u8; 16]) -> (usize, [u8; 16], bool) {
        let c = self.pair_cls[pair];
        let s1 = &self.syms[c.sym as usize];
        let t1 = if s1.plain {
            *tau
        } else {
            conj_edges(tau, &s1.sym)
        };
        if !c.nt {
            return (c.base as usize + prefix_rank(&t1), t1, false);
        }
        let k = self.pref_cls[c.base as usize + prefix_rank(&t1)];
        let s2 = &self.syms[(k & 31) as usize];
        let t2 = if s2.plain {
            t1
        } else {
            conj_edges(&t1, &s2.sym)
        };
        ((k >> 6) as usize, t2, k >> 5 & 1 == 1)
    }

    /// Table index of (pattern pair, tau, PLL bit) (one stabiliser variant; the table holds
    /// the same value in all of them).
    #[inline(always)]
    fn index_pair(&self, pair: usize, tau: &[u8; 16], pll: u32) -> usize {
        let (class, t, _) = self.class_of(pair, tau);
        (class * N_REST + rest_rank(&t)) * 2 + pll as usize
    }

    /// Table index from the full phase-2 centre coordinate and tau.
    #[inline(always)]
    pub fn index(&self, cc: u32, tau: &[u8; 16]) -> usize {
        self.index_pair(self.pair_of(cc), tau, cc & 1)
    }

    /// Table index of the x view of a state (U/D x R/L only).
    #[inline(always)]
    pub fn index_view(&self, cc: u32, tau: &[u8; 16]) -> usize {
        let v = self.view.as_ref().unwrap();
        let pair = v.pair[(cc / 24 % 70 * 12 + cc / 2 % 12) as usize] as usize;
        let t = if v.inverse {
            let mut inv = [0u8; 16];
            for e in 0..12 {
                inv[tau[e] as usize] = e as u8;
            }
            conj_edges(&inv, &v.sym)
        } else {
            conj_edges(tau, &v.sym)
        };
        self.index_pair(pair, &t, cc & 1)
    }

    /// The canonical index (minimum over the stabiliser), used by the generator.
    fn canonical(&self, pair: usize, tau: &[u8; 16], pll: u32) -> usize {
        let (class, t, nt) = self.class_of(pair, tau);
        let mut r = rest_rank(&t);
        if nt {
            let st = self.stab[class];
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

    /// The neighbour of (pattern pair, tau, PLL bit) by phase-2 move `k`.
    #[inline(always)]
    fn step(
        &self,
        p2: &P2Tables,
        (pair, tau, pll): (usize, &[u8; 16], u32),
        k: usize,
    ) -> (usize, [u8; 16], u32) {
        let nb = self.kind.nb();
        (
            self.a_move[pair / nb][k] as usize * nb + self.b_move[pair % nb][k] as usize,
            p2.tau_move(tau, k),
            pll ^ (WIDE >> k & 1),
        )
    }

    /// Exact distance of (pattern pair, tau, PLL bit), by walking down the table: each step
    /// to the first neighbour one closer.
    fn walk(&self, p2: &P2Tables, pair: usize, tau: &[u8; 16], pll: u32) -> u8 {
        let (mut pair, mut tau, mut pll) = (pair, *tau, pll);
        let mut v = self.get(self.index_pair(pair, &tau, pll));
        for d in 0..40 {
            let want = (v + 2) % 3;
            let next = (0..17)
                .map(|k| self.step(p2, (pair, &tau, pll), k))
                .find(|(p, t, l)| self.get(self.index_pair(*p, t, *l)) == want);
            match next {
                Some((p, t, l)) => {
                    (pair, tau, pll) = (p, t, l);
                    v = want;
                }
                None => return d,
            }
        }
        40
    }

    /// Exact distance of a state (full centre coordinate `cc`).
    pub fn distance(&self, p2: &P2Tables, cc: u32, tau: &[u8; 16]) -> u8 {
        self.walk(p2, self.pair_of(cc), tau, cc & 1)
    }

    /// Exact distance of the x view of a state (U/D x R/L only).
    pub fn distance_view(&self, p2: &P2Tables, cc: u32, tau: &[u8; 16]) -> u8 {
        // Walk in the view's own frame: the view of a state is a state of the table, and the
        // distance is a distance in the rotated (same) move graph.
        let v = self.view.as_ref().unwrap();
        let pair = v.pair[(cc / 24 % 70 * 12 + cc / 2 % 12) as usize] as usize;
        let t = if v.inverse {
            let mut inv = [0u8; 16];
            for e in 0..12 {
                inv[tau[e] as usize] = e as u8;
            }
            conj_edges(&inv, &v.sym)
        } else {
            conj_edges(tau, &v.sym)
        };
        self.walk(p2, pair, &t, cc & 1)
    }

    #[inline(always)]
    fn neighbour(&self, p2: &P2Tables, pair: usize, tau: &[u8; 16], pll: u32, k: usize) -> usize {
        let (p, t, l) = self.step(p2, (pair, tau, pll), k);
        self.canonical(p, &t, l)
    }

    /// The 17 neighbours' canonical indices of entry `idx`, xored (to time the generator's
    /// work per entry).
    #[doc(hidden)]
    pub fn probe(&self, p2: &P2Tables, idx: usize) -> usize {
        let (pair, tau, pll) = self.state(idx);
        (0..17).fold(0, |a, k| a ^ self.neighbour(p2, pair, &tau, pll, k))
    }

    /// The goal pairs: both patterns solved, either way round.
    fn goal_pairs(&self, p2: &P2Tables) -> Vec<usize> {
        let ob = self.kind.second();
        let mut out = vec![];
        for a in [0x0fu8, 0xf0] {
            for b in [0x0fu8, 0xf0] {
                out.push(Self::rank(p2, 0, a) * self.kind.nb() + Self::rank(p2, ob, b));
            }
        }
        out
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
        for g in self.goal_pairs(p2) {
            if set(self.canonical(g, &id, 0), 0) {
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
                                    let (pair, tau, pll) = self.state(i);
                                    for k in 0..17 {
                                        let j = self.neighbour(p2, pair, &tau, pll, k);
                                        if get(j) == 3 && set(j, next_code) {
                                            local += 1;
                                        }
                                    }
                                } else if v == 3 {
                                    let (pair, tau, pll) = self.state(i);
                                    for k in 0..17 {
                                        let j = self.neighbour(p2, pair, &tau, pll, k);
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
                    "  {:?} phase-2 table depth {d}: {frontier} ({count} of {n}, {:.1} s)",
                    self.kind,
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

    fn check_classes(kind: Kind) {
        let p2 = P2Tables::build_centres_for_tests();
        let b = Big::new(&p2, kind);
        eprintln!(
            "{kind:?}: {} syms, {} classes, {} entries, {:.2} GB",
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
            let (p3, t3, l3) = b.state(b.index_pair(pair, &tau, pll));
            assert_eq!(b.canonical(p3, &t3, l3), c);
        }
    }

    #[test]
    fn classes_and_symmetry_udfb() {
        check_classes(Kind::UdFb);
    }

    #[test]
    fn classes_and_symmetry_udrl() {
        check_classes(Kind::UdRl);
    }

    #[test]
    fn moves_and_views_match_cube() {
        // The pattern-pair moves agree with the cube, and the x view is the rotated cube.
        use crate::moves::Conj;
        let p2 = P2Tables::build_centres_for_tests();
        let bs = [Big::new(&p2, Kind::UdFb), Big::new(&p2, Kind::UdRl)];
        let m = *crate::cube::ROTATIONS
            .iter()
            .find(|m| m[0][0] == 1 && m[1][1] == 0)
            .unwrap();
        let cj = Conj::new(m);
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut c = crate::cube::Cube::SOLVED;
        for _ in 0..2000 {
            let i = rng.gen_range(0..17);
            let cc = p2.centre_coord_of(&c);
            let tau = p2.tau_of(&c);
            for b in &bs {
                let pair = b.pair_of(cc);
                let c2 = c.apply_perm(&MOVE_PERMS[P2_MOVES[i] as usize]);
                let cc2 = p2.centre_coord_of(&c2);
                let s = b.step(&p2, (pair, &tau, cc & 1), i);
                assert_eq!(s.0, b.pair_of(cc2));
                assert_eq!(s.2, cc2 & 1);
            }
            // The view: the rotated cube's own (U/D, R/L) pair and tau.
            let b = &bs[1];
            let y = cj.apply(&c);
            let mut masks = [0u8; 3];
            for o in 0..2 {
                for (k, &p) in POS[o].iter().enumerate() {
                    if y.ct[p] / 4 == FIRST[o] as u8 {
                        masks[o] |= 1 << k;
                    }
                }
            }
            let want = b.index_pair(
                Big::rank(&p2, 0, masks[0]) * 12 + Big::rank(&p2, 1, masks[1]),
                &p2.tau_of(&y),
                crate::phase2::pll_parity(&y, &p2.low),
            );
            let (pw, tw, lw) = b.state(want);
            let (pv, tv, lv) = b.state(b.index_view(cc, &tau));
            assert_eq!(b.canonical(pw, &tw, lw), b.canonical(pv, &tv, lv));
            c = c.apply_perm(&MOVE_PERMS[P2_MOVES[i] as usize]);
        }
    }

    fn reduced_bfs_matches_raw(kind: Kind, depth: u8) {
        let p2 = P2Tables::build_centres_for_tests();
        let b = Big::new(&p2, kind);
        let id: [u8; 16] = std::array::from_fn(|i| i as u8);
        let goals: Vec<(usize, [u8; 16], u32)> =
            b.goal_pairs(&p2).into_iter().map(|g| (g, id, 0)).collect();
        let step = |&(pair, tau, pll): &(usize, [u8; 16], u32), k: usize| {
            b.step(&p2, (pair, &tau, pll), k)
        };
        let mut raw = std::collections::HashMap::new();
        let mut fr = goals.clone();
        for g in &goals {
            raw.insert(*g, 0u8);
        }
        for d in 1..=depth {
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
        for d in 1..=depth {
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
        eprintln!("{kind:?}: {} raw states to depth {depth} agree", raw.len());
    }

    #[test]
    fn symmetry_reduced_bfs_matches_raw_udfb() {
        reduced_bfs_matches_raw(Kind::UdFb, 5);
    }

    #[test]
    fn symmetry_reduced_bfs_matches_raw_udrl() {
        reduced_bfs_matches_raw(Kind::UdRl, 5);
    }
}
