//! Phase 2: pair the dedges and solve the centres with U D F B, R2 L2, Uw2 Rw2 Fw2, in the
//! standard frame (R/L axis special) and with the colour pairs on their own axes.
//!
//! Coordinates:
//! * centres: the pattern of each colour pair on its 8 positions (70 x 70 x 12 allowed R/L
//!   patterns) and the PLL-parity bit (corner parity xor dedge parity, flipped only by the
//!   wide half turns): 117,600 values, exact table;
//! * the pairing permutation tau: for each edge, the edge holding the partner of the wing in
//!   its low slot (12!/2 even permutations), exact table, distance mod 3.

use crate::comb::{comb_rank, perm_rank, perm_unrank};
use crate::cube::{B, Cube, D, F, L, R, U};
use crate::moves::{ALLOWED_AFTER, Conj, MOVE_PERMS, N_MOVES, PHASE2_MASK, frame_rotation};
use crate::phase1::Frame;
use crate::table::{Mod3Table, UNSEEN};

pub const P2_MOVES: [u8; 17] = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 22, 25];
/// The wide half turns Uw2 Rw2 Fw2: the only moves that may end phase 2.
pub const WIDE_HALF: u32 = (1 << 19) | (1 << 22) | (1 << 25);
pub const N_CENTRE: usize = 70 * 70 * 12 * 2;
pub const N_TAU: usize = 239_500_800;

/// Centre positions of each orbit (face +, face -), in the cube's centre slot numbering.
const UD_POS: [usize; 8] = [0, 1, 2, 3, 12, 13, 14, 15];
const FB_POS: [usize; 8] = [8, 9, 10, 11, 20, 21, 22, 23];
const RL_POS: [usize; 8] = [4, 5, 6, 7, 16, 17, 18, 19];

pub struct P2Tables {
    /// Low wing slot of each edge.
    pub low: [u8; 12],
    rank70: [u8; 256],
    unrank70: [u8; 70],
    rl12: [u8; 256],
    rl12_inv: [u8; 12],
    /// Centre coordinate move table [coord][move index in P2_MOVES].
    pub centre_move: Vec<[u32; 17]>,
    pub centre_dist: Vec<u8>,
    /// Edge-slot permutations of the low and high slots under each phase-2 move:
    /// new_tau[e] = ph[tau[pl_inv[e]]].
    pub pl_inv: [[u8; 16]; 17],
    pub ph: [[u8; 16]; 17],
    pub tau_table: Mod3Table,
    /// Optional joint table over tau, the U/D pattern and the PLL bit (much tighter).
    pub joint: Option<crate::p2joint::Joint>,
    /// Optional large joint tables (tau, two centre patterns, the PLL bit; tighter still).
    pub bigs: Vec<crate::p2big::Big>,
    /// The bounds read from them: (table, through its x view).
    pub big_src: Vec<(usize, bool)>,
}

fn orbit_mask(c: &Cube, pos: &[usize; 8], colour: u8) -> u8 {
    let mut m = 0u8;
    for (i, &p) in pos.iter().enumerate() {
        if c.ct[p] / 4 == colour {
            m |= 1 << i;
        }
    }
    m
}

impl P2Tables {
    fn base() -> P2Tables {
        let fr = Frame::new(0);
        let mut low = [0u8; 12];
        let mut high = [0u8; 12];
        for e in 0..12 {
            let (a, b) = (2 * e as u8, 2 * e as u8 + 1);
            if fr.low_slots.contains(&a) {
                low[e] = a;
                high[e] = b;
            } else {
                low[e] = b;
                high[e] = a;
            }
        }
        let mut rank70 = [255u8; 256];
        let mut unrank70 = [0u8; 70];
        for m in 0u32..256 {
            if m.count_ones() == 4 {
                let r = comb_rank(m) as u8;
                rank70[m as usize] = r;
                unrank70[r as usize] = m as u8;
            }
        }
        let mut rl12 = [255u8; 256];
        let mut rl12_inv = [0u8; 12];
        let mut k = 0;
        for p in 0..256usize {
            if fr.rl_ok[p >> 6] >> (p & 63) & 1 == 1 {
                rl12[p] = k as u8;
                rl12_inv[k] = p as u8;
                k += 1;
            }
        }
        // fr.rl_pos for frame 0 is RL_POS (R face then L face, reading order).
        assert_eq!(fr.rl_pos.map(|x| x as usize), RL_POS);
        let mut pl_inv = [[0u8; 16]; 17];
        let mut ph = [[0u8; 16]; 17];
        for (i, &m) in P2_MOVES.iter().enumerate() {
            let src = &MOVE_PERMS[m as usize].wing_src;
            for e in 0..12 {
                // slot low[e] moves to slot t where src[t] == low[e]
                let tl = (0..24).find(|&t| src[t] == low[e]).unwrap();
                let th = (0..24).find(|&t| src[t] == high[e]).unwrap();
                let el = low
                    .iter()
                    .position(|&x| x as usize == tl)
                    .expect("low slot left the low set");
                let eh = high
                    .iter()
                    .position(|&x| x as usize == th)
                    .expect("high slot left the high set");
                pl_inv[i][el] = e as u8;
                ph[i][e] = eh as u8;
            }
            for e in 12..16 {
                pl_inv[i][e] = e as u8;
                ph[i][e] = e as u8;
            }
        }
        P2Tables {
            low,
            rank70,
            unrank70,
            rl12,
            rl12_inv,
            centre_move: vec![],
            centre_dist: vec![],
            pl_inv,
            ph,
            joint: None,
            bigs: vec![],
            big_src: vec![],
            tau_table: Mod3Table {
                data: crate::table::HugeBuf::zeroed(1),
            },
        }
    }

    /// Build all tables (the tau table from `tau_mod3` if given, else by BFS).
    pub fn build(tau_mod3: Option<Mod3Table>, threads: usize) -> P2Tables {
        let mut t = P2Tables::base();
        t.build_centres();
        t.tau_table = match tau_mod3 {
            Some(x) => x,
            None => Mod3Table::from_depths(&t.generate_tau(threads)),
        };
        t
    }

    /// Rank (0..12) of an allowed R/L pattern (bit i: position `RL_POS[i]` holds an R centre),
    /// 255 if not allowed.
    pub(crate) fn rank12(&self, mask: u8) -> u8 {
        self.rl12[mask as usize]
    }

    pub(crate) fn unrank12(&self, r: u8) -> u8 {
        self.rl12_inv[r as usize]
    }

    /// Add a large joint table and its bounds (directly, and through its view if it has one).
    pub fn add_big(&mut self, b: crate::p2big::Big) {
        let i = self.bigs.len();
        self.big_src.push((i, false));
        if b.view.is_some() {
            self.big_src.push((i, true));
        }
        assert!(self.big_src.len() <= MAX_BIG_SRC);
        self.bigs.push(b);
    }

    /// Table index of bound source `k` for a state.
    #[inline(always)]
    pub fn big_index(&self, k: usize, cc: u32, tau: &[u8; 16]) -> usize {
        let (i, view) = self.big_src[k];
        if view {
            self.bigs[i].index_view(cc, tau)
        } else {
            self.bigs[i].index(cc, tau)
        }
    }

    /// Exact distance of bound source `k` for a state.
    pub fn big_distance(&self, k: usize, cc: u32, tau: &[u8; 16]) -> u8 {
        let (i, view) = self.big_src[k];
        if view {
            self.bigs[i].distance_view(self, cc, tau)
        } else {
            self.bigs[i].distance(self, cc, tau)
        }
    }

    pub(crate) fn rank70(&self, mask: u8) -> u8 {
        self.rank70[mask as usize]
    }

    pub(crate) fn unrank70(&self, r: u8) -> u8 {
        self.unrank70[r as usize]
    }

    #[cfg(test)]
    pub(crate) fn base_for_tests() -> P2Tables {
        P2Tables::base()
    }

    #[cfg(test)]
    pub(crate) fn build_centres_for_tests() -> P2Tables {
        let mut t = P2Tables::base();
        t.build_centres();
        t
    }

    pub fn centre_coord_of(&self, c: &Cube) -> u32 {
        let ud = self.rank70[orbit_mask(c, &UD_POS, U as u8) as usize] as u32;
        let fb = self.rank70[orbit_mask(c, &FB_POS, F as u8) as usize] as u32;
        let rl = self.rl12[orbit_mask(c, &RL_POS, R as u8) as usize] as u32;
        assert!(
            ud < 70 && fb < 70 && rl < 12,
            "centres not staged for phase 2"
        );
        let pll = pll_parity(c, &self.low);
        ((ud * 70 + fb) * 12 + rl) * 2 + pll
    }

    pub(crate) fn centre_cube(&self, coord: u32) -> Cube {
        let pll = coord % 2;
        let rl = (coord / 2 % 12) as usize;
        let fb = (coord / 24 % 70) as usize;
        let ud = (coord / 24 / 70) as usize;
        let mut c = Cube::SOLVED;
        let mut put = |pos: &[usize; 8], mask: u8, a: usize, b: usize| {
            let (mut ia, mut ib) = (0u8, 0u8);
            for (i, &p) in pos.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    c.ct[p] = (a * 4) as u8 + ia;
                    ia += 1;
                } else {
                    c.ct[p] = (b * 4) as u8 + ib;
                    ib += 1;
                }
            }
        };
        put(&UD_POS, self.unrank70[ud], U, D);
        put(&FB_POS, self.unrank70[fb], F, B);
        put(&RL_POS, self.rl12_inv[rl], R, L);
        // Encode the PLL parity in the corners (one corner swap flips it).
        if pll == 1 {
            c.cp.swap(0, 1);
        }
        c
    }

    fn build_centres(&mut self) {
        let mut mv = vec![[0u32; 17]; N_CENTRE];
        for (coord, row) in mv.iter_mut().enumerate() {
            let c = self.centre_cube(coord as u32);
            for (i, &m) in P2_MOVES.iter().enumerate() {
                row[i] = self.centre_coord_of(&c.apply_perm(&MOVE_PERMS[m as usize]));
            }
        }
        // Goals: each orbit solved either way round, with an even number of pairs swapped
        // (a proper rotation of the colour scheme), and no PLL parity.
        let mut goals = vec![];
        for (sud, sfb, srl) in [(0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)] {
            let ud = self.rank70[if sud == 0 { 0x0f } else { 0xf0 }] as u32;
            let fb = self.rank70[if sfb == 0 { 0x0f } else { 0xf0 }] as u32;
            let rl = self.rl12[if srl == 0 { 0x0f } else { 0xf0 }] as u32;
            goals.push(((ud * 70 + fb) * 12 + rl) * 2);
        }
        let mut dist = vec![UNSEEN; N_CENTRE];
        let mut frontier: Vec<u32> = goals.clone();
        for &g in &goals {
            dist[g as usize] = 0;
        }
        let mut d = 0;
        while !frontier.is_empty() {
            let mut next = vec![];
            for &x in &frontier {
                for &y in &mv[x as usize] {
                    if dist[y as usize] == UNSEEN {
                        dist[y as usize] = d + 1;
                        next.push(y);
                    }
                }
            }
            frontier = next;
            d += 1;
        }
        self.centre_move = mv;
        self.centre_dist = dist;
    }

    pub fn tau_of(&self, c: &Cube) -> [u8; 16] {
        let mut slot_of = [0u8; 24];
        for s in 0..24 {
            slot_of[c.wp[s] as usize] = s as u8;
        }
        let mut tau = [0u8; 16];
        for e in 0..12 {
            let p = c.wp[self.low[e] as usize];
            tau[e] = slot_of[(p ^ 1) as usize] / 2;
        }
        for (e, t) in tau.iter_mut().enumerate().skip(12) {
            *t = e as u8;
        }
        tau
    }

    #[inline(always)]
    pub fn tau_move(&self, tau: &[u8; 16], i: usize) -> [u8; 16] {
        let mut out = [0u8; 16];
        let pl = &self.pl_inv[i];
        let ph = &self.ph[i];
        for e in 0..16 {
            out[e] = ph[tau[pl[e] as usize] as usize];
        }
        out
    }

    #[inline(always)]
    pub fn tau_rank(tau: &[u8; 16]) -> u64 {
        perm_rank(&tau[..12]) / 2
    }

    fn generate_tau(&self, threads: usize) -> Vec<u8> {
        let id: [u8; 16] = std::array::from_fn(|i| i as u8);
        crate::table::bfs(N_TAU, &[Self::tau_rank(&id)], threads, |r, f| {
            let tau = tau_unrank(r);
            for i in 0..17 {
                f(Self::tau_rank(&self.tau_move(&tau, i)));
            }
        })
    }

    pub fn tau_value(&self, tau: &[u8; 16]) -> u8 {
        self.tau_table.get(Self::tau_rank(tau))
    }

    /// Exact tau distance by walking down the mod-3 table.
    pub fn tau_distance(&self, tau: &[u8; 16]) -> u8 {
        let mut t = *tau;
        let mut d = 0;
        loop {
            let v = self.tau_value(&t);
            match (0..17)
                .map(|i| self.tau_move(&t, i))
                .find(|c| self.tau_value(c) == (v + 2) % 3)
            {
                Some(c) => {
                    t = c;
                    d += 1;
                }
                None => return d,
            }
        }
    }
}

/// The even permutation of 12 with rank `r` (rank = Lehmer rank / 2).
pub fn tau_unrank(r: u64) -> [u8; 16] {
    let mut p = perm_unrank(r * 2, 12);
    if crate::comb::perm_parity(&p) == 1 {
        p.swap(10, 11);
    }
    let mut out = [0u8; 16];
    out[..12].copy_from_slice(&p);
    for (e, t) in out.iter_mut().enumerate().skip(12) {
        *t = e as u8;
    }
    out
}

/// Corner parity xor dedge parity (dedges read through the wings in the low slots).
pub fn pll_parity(c: &Cube, low: &[u8; 12]) -> u32 {
    let mut ep = [0u8; 12];
    for e in 0..12 {
        ep[e] = c.wp[low[e] as usize] / 2;
    }
    (Cube::parity(&c.cp) ^ Cube::parity(&ep)) as u32
}

/// Bring a phase-1 endpoint into the standard frame and colour assignment: conjugate by the
/// frame rotation, then recolour so that the colour pair on each axis is that axis's own.
pub fn standardise(c: &Cube, frame: usize) -> Cube {
    let rot = frame_rotation(frame);
    let x = Conj::new(rot).apply(c);
    // Colour pair (0: U/D, 1: R/L, 2: F/B) on each axis.
    let pair = |colour: u8| -> usize {
        match colour as usize {
            U | D => 0,
            R | L => 1,
            _ => 2,
        }
    };
    let on_ud = pair(x.ct[0] / 4);
    let on_rl = pair(x.ct[4] / 4);
    let on_fb = pair(x.ct[8] / 4);
    let face_pair = [0usize, 1, 2, 0, 1, 2]; // pair of each face U R F D L B
    for m in crate::cube::ROTATIONS.iter() {
        let cj = Conj::new(*m);
        let map = cj.colour_map();
        let ok = |p: usize, target: usize| {
            (0..6)
                .filter(|&f| face_pair[f] == p)
                .all(|f| face_pair[map[f] as usize] == target)
        };
        if ok(on_ud, 0) && ok(on_rl, 1) && ok(on_fb, 2) {
            return cj.recolour(&x);
        }
    }
    unreachable!("no rotation standardises the colour pairs")
}

/// Moves allowed to open phase 2 after the phase-1 moves `moves1` (already in the phase-2
/// frame): any move on a face not used by phase 1's trailing run of same-axis moves. Moves of
/// that axis may come in any order relative to the run: "P Uw | U ..." is the reduction
/// "P U Uw | ...", whose endpoint phase 1 does not report for frames where U is a phase-2
/// move (see `phase1::trailing_run`), so each reduction is produced exactly once.
pub fn first_moves(moves1: &[u8]) -> u32 {
    let mut faces = 0u32;
    if let Some(&last) = moves1.last() {
        let axis = (last / 3) % 3;
        for &m in moves1.iter().rev().take_while(|&&m| (m / 3) % 3 == axis) {
            faces |= 0b111 << (3 * (m as usize / 3));
        }
    }
    ALLOWED_AFTER[N_MOVES] & !faces
}

/// The moves of each axis (U/D, R/L, F/B), a bit per move index.
const AXIS_MOVES: [u32; 3] = [0x1c0e07, 0xe07038, 0x70381c0];

/// Moves allowed after move `m` at a node whose own moves were restricted to `after`: the
/// usual canonical order, and a restriction on `m`'s axis carries on along a run of moves of
/// that axis (so phase 1's trailing run keeps its faces excluded until another axis).
#[inline(always)]
fn after_child(after: u32, m: usize) -> u32 {
    ALLOWED_AFTER[m] & (after | !AXIS_MOVES[(m / 3) % 3])
}

/// Moves that may end phase 2 after move `last`: a wide half turn, but not one right after an
/// outer turn on its own axis ("U Uw2" = "Uw2 U": the shorter reduction, with U left to
/// phase 3, covers it).
#[inline(always)]
fn last_mask(last: usize) -> u32 {
    if last < 18 {
        WIDE_HALF & !(1 << (19 + 3 * ((last / 3) % 3)))
    } else {
        WIDE_HALF
    }
}

/// Depth-first phase-2 search for solutions of exactly `left` moves.
pub struct Search2<'a> {
    pub t: &'a P2Tables,
    pub path: Vec<u8>,
    pub nodes: u64,
    pub abort: Option<&'a std::sync::atomic::AtomicBool>,
}

impl<'a> Search2<'a> {
    pub fn new(t: &'a P2Tables) -> Self {
        Search2 {
            t,
            path: Vec::with_capacity(24),
            nodes: 0,
            abort: None,
        }
    }

    /// Explore below (centre coord, tau with exact tau distance `dt`); `last` is the previous
    /// move (27-move index, `N_MOVES` for none) and `after` the moves allowed next
    /// (`ALLOWED_AFTER[last]`, or at the root the moves allowed after phase 1, see
    /// [`first_moves`]). Calls `found` for each solution; stops when it returns false. Returns
    /// false if stopped.
    #[allow(clippy::too_many_arguments)]
    pub fn dfs(
        &mut self,
        cc: u32,
        tau: &[u8; 16],
        dt: u8,
        left: u8,
        last: usize,
        after: u32,
        found: &mut dyn FnMut(&[u8]) -> bool,
    ) -> bool {
        if left == 0 {
            if cc_is_goal(self.t, cc) && dt == 0 {
                return found(&self.path);
            }
            return true;
        }
        let allowed = after & PHASE2_MASK & if left == 1 { last_mask(last) } else { u32::MAX };
        for (i, &m) in P2_MOVES.iter().enumerate() {
            if allowed >> m & 1 == 0 {
                continue;
            }
            self.nodes += 1;
            if self.nodes & 4095 == 0
                && self
                    .abort
                    .is_some_and(|a| a.load(std::sync::atomic::Ordering::Relaxed))
            {
                return false;
            }
            let c2 = self.t.centre_move[cc as usize][i];
            if self.t.centre_dist[c2 as usize] >= left {
                continue;
            }
            let t2 = self.t.tau_move(tau, i);
            let d2 = Mod3Table::decode(dt, self.t.tau_value(&t2));
            if d2 >= left {
                continue;
            }
            self.path.push(m);
            let go = self.dfs(
                c2,
                &t2,
                d2,
                left - 1,
                m as usize,
                after_child(after, m as usize),
                found,
            );
            self.path.pop();
            if !go {
                return false;
            }
        }
        true
    }
}

impl<'a> Search2<'a> {
    /// [`Search2::dfs`] with the joint table: `dj` and `df` are the exact joint distances of
    /// the state and of its F/B view (the table read for the F/B centre pattern; the two
    /// bounds together expand ~7x fewer nodes than the U/D one alone at deep levels).
    #[allow(clippy::too_many_arguments)]
    pub fn dfs_joint(
        &mut self,
        cc: u32,
        tau: &[u8; 16],
        dj: u8,
        df: u8,
        left: u8,
        last: usize,
        after: u32,
        found: &mut dyn FnMut(&[u8]) -> bool,
    ) -> bool {
        if left == 0 {
            if cc_is_goal(self.t, cc) && dj == 0 {
                return found(&self.path);
            }
            return true;
        }
        let t = self.t;
        let j = t.joint.as_ref().unwrap();
        let allowed = after & PHASE2_MASK & if left == 1 { last_mask(last) } else { u32::MAX };
        // Stage 1: children with the centre bound, joint entries (both views) prefetched.
        let mut kids = [(0u32, [0u8; 16], 0usize, 0usize, 0u8); 17];
        let mut nk = 0;
        for (i, &m) in P2_MOVES.iter().enumerate() {
            if allowed >> m & 1 == 0 {
                continue;
            }
            let c2 = t.centre_move[cc as usize][i];
            if t.centre_dist[c2 as usize] >= left {
                continue;
            }
            let t2 = t.tau_move(tau, i);
            let idx = j.index(c2 / 1680, &t2, c2 & 1);
            j.prefetch(idx);
            let idf = j.index_fb(c2, &t2);
            j.prefetch(idf);
            kids[nk] = (c2, t2, idx, idf, m);
            nk += 1;
        }
        self.nodes += nk as u64;
        if self
            .abort
            .is_some_and(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        {
            return false;
        }
        // Stage 2: read, decode, descend.
        for k in 0..nk {
            let (c2, t2, idx, idf, m) = kids[k];
            let d2 = Mod3Table::decode(dj, j.get(idx));
            if d2 >= left {
                continue;
            }
            let f2 = Mod3Table::decode(df, j.get(idf));
            if f2 >= left {
                continue;
            }
            self.path.push(m);
            let go = self.dfs_joint(
                c2,
                &t2,
                d2,
                f2,
                left - 1,
                m as usize,
                after_child(after, m as usize),
                found,
            );
            self.path.pop();
            if !go {
                return false;
            }
        }
        true
    }
}

/// At most this many bounds from the large joint tables (U/D x R/L with its view, U/D x F/B).
pub const MAX_BIG_SRC: usize = 3;

impl<'a> Search2<'a> {
    /// [`Search2::dfs`] with the large joint tables: `ds[k]` is the exact distance from bound
    /// source `k` (see [`P2Tables::big_src`]).
    #[allow(clippy::too_many_arguments)]
    pub fn dfs_big(
        &mut self,
        cc: u32,
        tau: &[u8; 16],
        ds: [u8; MAX_BIG_SRC],
        left: u8,
        last: usize,
        after: u32,
        found: &mut dyn FnMut(&[u8]) -> bool,
    ) -> bool {
        let t = self.t;
        let ns = t.big_src.len();
        if left == 0 {
            if cc_is_goal(t, cc) && ds[..ns].iter().all(|&d| d == 0) {
                return found(&self.path);
            }
            return true;
        }
        let allowed = after & PHASE2_MASK & if left == 1 { last_mask(last) } else { u32::MAX };
        // Stage 1: children with the centre bound, table entries prefetched.
        let mut kids = [(0u32, [0u8; 16], [0usize; MAX_BIG_SRC], 0u8); 17];
        let mut nk = 0;
        for (i, &m) in P2_MOVES.iter().enumerate() {
            if allowed >> m & 1 == 0 {
                continue;
            }
            let c2 = t.centre_move[cc as usize][i];
            if t.centre_dist[c2 as usize] >= left {
                continue;
            }
            let t2 = t.tau_move(tau, i);
            let mut idx = [0usize; MAX_BIG_SRC];
            for k in 0..ns {
                idx[k] = t.big_index(k, c2, &t2);
                t.bigs[t.big_src[k].0].prefetch(idx[k]);
            }
            kids[nk] = (c2, t2, idx, m);
            nk += 1;
        }
        self.nodes += nk as u64;
        if self
            .abort
            .is_some_and(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        {
            return false;
        }
        // Stage 2: read, decode, descend.
        'kid: for k in 0..nk {
            let (c2, t2, idx, m) = kids[k];
            let mut d2 = [0u8; MAX_BIG_SRC];
            for s in 0..ns {
                d2[s] = Mod3Table::decode(ds[s], t.bigs[t.big_src[s].0].get(idx[s]));
                if d2[s] >= left {
                    continue 'kid;
                }
            }
            self.path.push(m);
            let go = self.dfs_big(
                c2,
                &t2,
                d2,
                left - 1,
                m as usize,
                after_child(after, m as usize),
                found,
            );
            self.path.pop();
            if !go {
                return false;
            }
        }
        true
    }
}

#[inline(always)]
fn cc_is_goal(t: &P2Tables, cc: u32) -> bool {
    t.centre_dist[cc as usize] == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_rank_roundtrip() {
        for r in [0u64, 1, 12345, N_TAU as u64 - 1] {
            assert_eq!(P2Tables::tau_rank(&tau_unrank(r)), r);
        }
    }

    #[test]
    fn tau_moves_match_cube() {
        let t = P2Tables::base();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        // A random phase-2 position: apply random phase-2 moves to a pairable cube.
        let mut c = Cube::SOLVED;
        for _ in 0..40 {
            let i = rng.gen_range(0..17);
            let tau = t.tau_of(&c);
            c = c.apply_perm(&MOVE_PERMS[P2_MOVES[i] as usize]);
            assert_eq!(t.tau_move(&tau, i), t.tau_of(&c));
        }
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    #[test]
    fn axis_moves_and_after_child() {
        for m in 0..N_MOVES {
            assert!(AXIS_MOVES[(m / 3) % 3] >> m & 1 == 1);
            assert_eq!(AXIS_MOVES.iter().filter(|&&a| a >> m & 1 == 1).count(), 1);
            // Without a phase-1 run, the usual canonical order.
            for last in 0..=N_MOVES {
                if ALLOWED_AFTER[last] >> m & 1 == 1 {
                    assert_eq!(after_child(ALLOWED_AFTER[last], m), ALLOWED_AFTER[m]);
                }
            }
        }
    }
}
