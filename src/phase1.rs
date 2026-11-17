//! Phase 1: stage the centres and make the wings pairable.
//!
//! Goal (for at least one of three frames, the frame naming the axis that phase 2 turns only by
//! half turns, and with any assignment of the three colour pairs to the three axes):
//! * every centre is on a face of the axis its colour pair was assigned to;
//! * the 8 centres on the faces of the frame's axis form one of the patterns phase 2 can solve;
//! * the wing permutation is even;
//! * the frame's 12 "low" wing slots hold exactly one wing of each dedge, and an even number of
//!   them are "high" wings.
//!
//! The search is guided by an exact table of the distance to staged centres (any assignment).
//! That distance is the same with all 36 outer block moves as with the solver's 27, so it is
//! invariant under the 48 symmetries; with the colour pairs unlabelled except for U/D, the
//! table has 15582 x 6435 entries (25 MB at two bits).

use crate::comb::{BINOM, binom, comb_rank, comb_unrank};
use crate::cube::{Cube, GEO, ROTATIONS, centre_slot, mat_apply};
use crate::moves::{
    ALLOWED_AFTER, MOVE_PERMS, N_MOVES, PHASE2_MASK, WIDE_QUARTER, frame_rotation, rotate_move,
};
use crate::simd::V32;
use crate::table::Mod3Table;

pub const N_CLASSES_A: usize = 15582;
pub const N_SPLIT: usize = 6435;
const MASK24: u32 = 0xff_ffff;

/// Centre colour codes: the colour pair (class) is code >> 1, U/D = 0, R/L = 1, F/B = 2.
pub const CODE_OF_FACE: [u8; 6] = [0, 2, 4, 1, 3, 5];
#[inline(always)]
pub fn pext(x: u32, m: u32) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        std::arch::x86_64::_pext_u32(x, m)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut r = 0;
        let mut k = 0;
        let mut m = m;
        while m != 0 {
            let b = m & m.wrapping_neg();
            if x & b != 0 {
                r |= 1 << k;
            }
            k += 1;
            m &= m - 1;
        }
        r
    }
}

#[inline(always)]
pub fn pdep(x: u32, m: u32) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        std::arch::x86_64::_pdep_u32(x, m)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut r = 0;
        let mut k = 0;
        let mut m = m;
        while m != 0 {
            let b = m & m.wrapping_neg();
            if x >> k & 1 != 0 {
                r |= b;
            }
            k += 1;
            m &= m - 1;
        }
        r
    }
}

/// Static data for the staging coordinate.
pub struct Staging {
    /// For each rank of an 8-of-24 mask: class << 8 | symmetry taking it to the class rep.
    a_info: crate::table::HugeBuf<u32>,
    class_rep: Vec<u32>,
    /// Non-trivial stabiliser symmetries of each class (empty for most).
    stab: Vec<Vec<u8>>,
    /// Mask permutation tables: sym_perm[s][byte][value].
    sym_perm: Vec<[[u32; 256]; 3]>,
    move_perm: Vec<[[u32; 256]; 3]>,
    /// Unlabelled rank of an 8-of-16 mask (x and its complement share a rank).
    unl: Vec<u16>,
    unl_inv: Vec<u16>,
    /// Byte-sliced colex rank of 8-of-24 masks.
    rank_t: [[[u32; 256]; 9]; 3],
}

fn perm_tables(perm: &[u8; 24]) -> [[u32; 256]; 3] {
    let mut t = [[0u32; 256]; 3];
    for (k, tk) in t.iter_mut().enumerate() {
        for (v, e) in tk.iter_mut().enumerate() {
            let mut out = 0;
            for j in 0..8 {
                if v >> j & 1 == 1 {
                    out |= 1 << perm[8 * k + j];
                }
            }
            *e = out;
        }
    }
    t
}

#[inline(always)]
fn apply_mask(t: &[[u32; 256]; 3], m: u32) -> u32 {
    t[0][(m & 255) as usize] | t[1][(m >> 8 & 255) as usize] | t[2][(m >> 16) as usize]
}

/// The 48 symmetries as centre-slot permutations (24 rotations, then the same composed with
/// the central inversion).
pub fn centre_syms() -> Vec<[u8; 24]> {
    let g = &*GEO;
    let mut out = vec![];
    for inv in [1i8, -1] {
        for m in ROTATIONS.iter() {
            let mut p = [0u8; 24];
            for (i, e) in p.iter_mut().enumerate() {
                let v = mat_apply(m, g.centre_pos[i]).map(|c| c * inv);
                *e = centre_slot(v) as u8;
            }
            out.push(p);
        }
    }
    out
}

impl Staging {
    pub fn new() -> Staging {
        let syms = centre_syms();
        let sym_perm: Vec<_> = syms.iter().map(perm_tables).collect();
        let move_perm: Vec<_> = (0..N_MOVES)
            .map(|m| {
                // Mask of positions moves: position s goes to t where src[t] = s.
                let src = &MOVE_PERMS[m].centre_src;
                let mut dst = [0u8; 24];
                for t in 0..24 {
                    dst[src[t] as usize] = t as u8;
                }
                perm_tables(&dst)
            })
            .collect();
        let mut rank_t = [[[0u32; 256]; 9]; 3];
        for k in 0..3 {
            for c in 0..9 {
                for v in 0..256usize {
                    let mut r = 0;
                    let mut t = 0;
                    for j in 0..8 {
                        if v >> j & 1 == 1 {
                            if c + t < 8 {
                                r += BINOM[8 * k + j][c + t + 1];
                            }
                            t += 1;
                        }
                    }
                    rank_t[k][c][v] = r;
                }
            }
        }
        let n_a = binom(24, 8) as usize;
        let mut a_info = vec![u32::MAX; n_a];
        let mut class_rep = vec![];
        let mut stab = vec![];
        for r in 0..n_a {
            if a_info[r] != u32::MAX {
                continue;
            }
            let rep = comb_unrank(r as u32, 24, 8);
            let class = class_rep.len() as u32;
            class_rep.push(rep);
            let mut st = vec![];
            for (s, sp) in sym_perm.iter().enumerate() {
                let img = apply_mask(sp, rep);
                if img == rep && s != 0 {
                    st.push(s as u8);
                }
                let ri = comb_rank(img) as usize;
                if a_info[ri] == u32::MAX {
                    // sym taking img back to rep: the inverse of s.
                    let inv = (0..48)
                        .find(|&t| apply_mask(&sym_perm[t], img) == rep)
                        .unwrap();
                    a_info[ri] = class << 8 | inv as u32;
                }
            }
            stab.push(st);
        }
        assert_eq!(class_rep.len(), N_CLASSES_A);
        let mut unl = vec![u16::MAX; 1 << 16];
        let mut unl_inv = vec![];
        for x in 0u32..(1 << 16) {
            if x.count_ones() == 8 && unl[x as usize] == u16::MAX {
                let id = unl_inv.len() as u16;
                unl[x as usize] = id;
                unl[(!x & 0xffff) as usize] = id;
                unl_inv.push(x as u16);
            }
        }
        assert_eq!(unl_inv.len(), N_SPLIT);
        Staging {
            a_info: {
                // Huge pages: this table is read at random for every child.
                let mut h = crate::table::HugeBuf::<u32>::zeroed(a_info.len());
                h.as_mut_slice().copy_from_slice(&a_info);
                h
            },
            class_rep,
            stab,
            sym_perm,
            move_perm,
            unl,
            unl_inv,
            rank_t,
        }
    }

    #[inline(always)]
    pub fn rank24(&self, m: u32) -> u32 {
        let b0 = m & 255;
        let b1 = m >> 8 & 255;
        let b2 = m >> 16;
        let c0 = b0.count_ones() as usize;
        let c1 = c0 + b1.count_ones() as usize;
        self.rank_t[0][0][b0 as usize]
            + self.rank_t[1][c0][b1 as usize]
            + self.rank_t[2][c1][b2 as usize]
    }

    /// Canonical table index of the centre classes given the U/D-colour and R/L-colour
    /// position masks (the minimum over the class's stabiliser).
    pub fn index(&self, ud: u32, rl: u32) -> u64 {
        let info = self.a_info[self.rank24(ud) as usize];
        let class = (info >> 8) as usize;
        let s = (info & 255) as usize;
        let rem = !self.class_rep[class] & MASK24;
        let b = apply_mask(&self.sym_perm[s], rl);
        let mut u = self.unl[pext(b, rem) as usize];
        for &t in &self.stab[class] {
            let bt = apply_mask(&self.sym_perm[t as usize], b);
            u = u.min(self.unl[pext(bt, rem) as usize]);
        }
        (class * N_SPLIT + u as usize) as u64
    }

    /// The class info (class << 8 | symmetry to the representative) of a U/D mask; it is the
    /// first, separately prefetchable, step of [`Staging::fast_index`].
    #[inline(always)]
    pub fn info(&self, ud: u32) -> *const u32 {
        unsafe { self.a_info.as_ptr().add(self.rank24(ud) as usize) }
    }

    /// A table index of the centre classes, without minimising over the stabiliser: the
    /// table holds the same value at every stabiliser variant.
    #[inline(always)]
    pub fn fast_index(&self, info: u32, rl: u32) -> u64 {
        let class = (info >> 8) as usize;
        let s = (info & 255) as usize;
        let b = apply_mask(&self.sym_perm[s], rl);
        let rem = !self.class_rep[class] & MASK24;
        (class * N_SPLIT + self.unl[pext(b, rem) as usize] as usize) as u64
    }

    pub fn rep_of(&self, idx: u64) -> (u32, u32) {
        let class = idx as usize / N_SPLIT;
        let u = idx as usize % N_SPLIT;
        let rep = self.class_rep[class];
        let rem = !rep & MASK24;
        (rep, pdep(self.unl_inv[u] as u32, rem))
    }

    /// Index into the large table (no spatial symmetry): the U/D mask's rank, and the
    /// unlabelled split of the other 16 positions between the R/L and F/B colours.
    #[inline(always)]
    pub fn big_index(&self, ud: u32, rl: u32) -> u64 {
        self.rank24(ud) as u64 * N_SPLIT as u64 + self.unl[pext(rl, !ud & MASK24) as usize] as u64
    }

    pub fn big_len() -> usize {
        735_471 * N_SPLIT
    }

    /// Generate the large table (distance to staged centres, any assignment).
    pub fn generate_big(&self, threads: usize) -> Vec<u8> {
        let n = Self::big_len();
        let uds: Vec<u32> = (0..735_471).map(|r| comb_unrank(r, 24, 8)).collect();
        let ax = Staging::axis_masks();
        let mut goals: Vec<u64> = vec![];
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    goals.push(self.big_index(ax[a], ax[b]));
                }
            }
        }
        goals.sort();
        goals.dedup();
        crate::table::bfs(n, &goals, threads, |i, f| {
            let ud = uds[i as usize / N_SPLIT];
            let rl = pdep(self.unl_inv[i as usize % N_SPLIT] as u32, !ud & MASK24);
            for m in 0..N_MOVES {
                let t = &self.move_perm[m];
                f(self.big_index(apply_mask(t, ud), apply_mask(t, rl)));
            }
        })
    }

    /// Face masks of the three axes (U+D, R+L, F+B centre positions).
    pub fn axis_masks() -> [u32; 3] {
        [0xf00f, 0xf00f0, 0xf00f00].map(|m: u32| m & MASK24)
    }

    pub fn goals(&self) -> Vec<u64> {
        let ax = Staging::axis_masks();
        let mut g = vec![];
        for a in 0..3 {
            for b in 0..3 {
                if a != b {
                    g.push(self.index(ax[a], ax[b]));
                }
            }
        }
        g.sort();
        g.dedup();
        g
    }

    pub fn generate(&self, threads: usize) -> Vec<u8> {
        let n = N_CLASSES_A * N_SPLIT;
        let mut depths = crate::table::bfs(n, &self.goals(), threads, |i, f| {
            let (ud, rl) = self.rep_of(i);
            for m in 0..N_MOVES {
                let t = &self.move_perm[m];
                f(self.index(apply_mask(t, ud), apply_mask(t, rl)));
            }
        });
        // Give every non-canonical entry (another stabiliser variant of a canonical one) the
        // canonical entry's value. (The BFS's backward passes also visit non-canonical
        // entries, whose depths are not reliable.)
        let fix = std::sync::Mutex::new(vec![]);
        crate::par::for_chunks(n, 1 << 16, |r| {
            let mut local = vec![];
            for i in r {
                let (ud, rl) = self.rep_of(i as u64);
                let j = self.index(ud, rl) as usize;
                if j != i {
                    local.push((i, j));
                }
            }
            fix.lock().unwrap().extend(local);
        });
        for (i, j) in fix.into_inner().unwrap() {
            depths[i] = depths[j];
        }
        depths
    }
}

impl Default for Staging {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-frame endpoint conditions.
pub struct Frame {
    /// Moves of phase 2 in this frame.
    pub p2_mask: u32,
    /// The 12 low wing slots.
    pub low_slots: [u8; 12],
    /// Pieces whose home slot is high.
    pub high_pieces: u32,
    /// Centre positions of the special axis's two faces (8).
    pub rl_pos: [u8; 8],
    /// Allowed patterns (bit i = position rl_pos[i] holds the pair's first colour).
    pub rl_ok: [u64; 4],
    /// 0x80 at the low slots (looked up by each wing's slot).
    pub low_tab: V32,
}

impl Frame {
    pub fn new(f: usize) -> Frame {
        let rot = frame_rotation(f);
        let mut p2_mask = 0u32;
        for m in 0..N_MOVES {
            if PHASE2_MASK >> rotate_move(m, &rot) & 1 == 1 {
                p2_mask |= 1 << m;
            }
        }
        // Low slots: the orbit of wing slot 0 under the phase-2 moves.
        let mut low = [false; 24];
        low[0] = true;
        let mut changed = true;
        while changed {
            changed = false;
            for m in 0..N_MOVES {
                if p2_mask >> m & 1 == 0 {
                    continue;
                }
                let src = &MOVE_PERMS[m].wing_src;
                for t in 0..24 {
                    if low[src[t] as usize] && !low[t] {
                        low[t] = true;
                        changed = true;
                    }
                }
            }
        }
        let low_slots: Vec<u8> = (0..24).filter(|&s| low[s]).map(|s| s as u8).collect();
        assert_eq!(low_slots.len(), 12);
        let mut high_pieces = 0u32;
        for s in 0..24 {
            if !low[s] {
                high_pieces |= 1 << s;
            }
        }
        // Centre positions on the faces of the special axis: those mapped by the frame
        // rotation onto the R and L faces.
        let g = &*GEO;
        let mut rl_pos = [0u8; 8];
        let mut k = 0;
        for face_pos in [4usize, 16] {
            for j in 0..4 {
                let target = g.centre_pos[face_pos + j];
                let s = (0..24)
                    .find(|&s| mat_apply(&rot, g.centre_pos[s]) == target)
                    .unwrap();
                rl_pos[k] = s as u8;
                k += 1;
            }
        }
        // Allowed patterns: closure of the two solved patterns under the phase-2 moves.
        let mut ok = [0u64; 4];
        let mut stack = vec![0x0fu8, 0xf0];
        for &p in &stack {
            ok[p as usize >> 6] |= 1 << (p & 63);
        }
        while let Some(p) = stack.pop() {
            // Expand to a 24-position colour mask, move, contract.
            let mut full = 0u32;
            for i in 0..8 {
                if p >> i & 1 == 1 {
                    full |= 1 << rl_pos[i];
                }
            }
            for m in 0..N_MOVES {
                if p2_mask >> m & 1 == 0 {
                    continue;
                }
                let src = &MOVE_PERMS[m].centre_src;
                let mut moved = 0u32;
                for t in 0..24 {
                    if full >> src[t] & 1 == 1 {
                        moved |= 1 << t;
                    }
                }
                let mut q = 0u8;
                for i in 0..8 {
                    if moved >> rl_pos[i] & 1 == 1 {
                        q |= 1 << i;
                    }
                }
                assert_eq!(q.count_ones(), 4, "special-axis centres left their faces");
                if ok[q as usize >> 6] >> (q & 63) & 1 == 0 {
                    ok[q as usize >> 6] |= 1 << (q & 63);
                    stack.push(q);
                }
            }
        }
        let count: u32 = ok.iter().map(|w| w.count_ones()).sum();
        assert_eq!(count, 12);
        let low_slots: [u8; 12] = low_slots.try_into().unwrap();
        let mut low_tab = V32::ZERO;
        for &sl in &low_slots {
            low_tab.0[sl as usize] = 0x80;
        }
        Frame {
            p2_mask,
            low_tab,
            low_slots,
            high_pieces,
            rl_pos,
            rl_ok: ok,
        }
    }

    /// Are the wings pairable in this frame (one wing of each dedge in the low slots, an even
    /// number of high wings among them)? `pos` holds the slot of each wing.
    #[inline(always)]
    pub fn wings_ok(&self, pos: &V32) -> bool {
        let m = crate::simd::movemask(&self.low_tab.gather(pos)) & MASK24;
        ((m ^ (m >> 1)) & 0x55_5555) == 0x55_5555
            && (m & self.high_pieces).count_ones().is_multiple_of(2)
    }

    /// Is the special axis's centre pattern one phase 2 can solve (assuming staged centres)?
    #[inline(always)]
    pub fn centres_ok(&self, centres: &V32) -> bool {
        let mut p = 0u8;
        for i in 0..8 {
            p |= (centres.0[self.rl_pos[i] as usize] & 1) << i;
        }
        self.rl_ok[p as usize >> 6] >> (p & 63) & 1 == 1
    }
}

/// Phase-1 search state.
#[derive(Clone, Copy)]
pub struct P1State {
    pub centres: V32,
    /// The slot of each wing piece.
    pub pos: V32,
    pub parity: bool,
}

impl P1State {
    pub fn from_cube(c: &Cube) -> P1State {
        let mut centres = V32::ZERO;
        for s in 0..24 {
            centres.0[s] = CODE_OF_FACE[(c.ct[s] / 4) as usize];
        }
        let mut pos = V32::ZERO;
        for sl in 0..24 {
            pos.0[c.wp[sl] as usize] = sl as u8;
        }
        P1State {
            centres,
            pos,
            parity: Cube::parity(&c.wp) == 1,
        }
    }

    #[inline(always)]
    pub fn apply(&self, m: usize, t: &P1Tables) -> P1State {
        P1State {
            centres: self.centres.gather(&t.centre_idx[m]),
            pos: t.wing_dst[m].gather(&self.pos),
            parity: self.parity ^ (WIDE_QUARTER >> m & 1 == 1),
        }
    }

    /// The state after `moves` and its exact staging distance, from this state at exact
    /// distance `d`.
    pub fn after(&self, moves: &[u8], d: u8, t: &P1Tables) -> (P1State, u8) {
        moves.iter().fold((*self, d), |(s, d), &m| {
            let c = s.apply(m as usize, t);
            (c, Mod3Table::decode(d, t.value(&c)))
        })
    }

    /// The U/D-colour and R/L-colour position masks of the centres.
    #[inline(always)]
    pub fn masks(&self) -> (u32, u32) {
        centre_masks(self.centres)
    }

    /// The centre masks of the child after move `m` (those of `self.apply(m, t)`).
    #[inline(always)]
    pub fn child_masks(&self, m: usize, t: &P1Tables) -> (u32, u32) {
        centre_masks(self.centres.gather(&t.centre_idx[m]))
    }
}

#[inline(always)]
fn centre_masks(centres: V32) -> (u32, u32) {
    let rl = centres.bit_mask(1) & MASK24;
    let fb = centres.bit_mask(2) & MASK24;
    (!(rl | fb) & MASK24, rl)
}

pub struct P1Tables {
    pub staging: Staging,
    pub table: Mod3Table,
    /// The table is the large one (no spatial symmetry reduction).
    pub big: bool,
    pub frames: [Frame; 3],
    pub centre_idx: [V32; N_MOVES],
    /// Destination slot of each slot under each move (applied to wing positions).
    pub wing_dst: [V32; N_MOVES],
}

impl P1Tables {
    pub fn with_table(staging: Staging, table: Mod3Table) -> P1Tables {
        let frames = [Frame::new(0), Frame::new(1), Frame::new(2)];
        let centre_idx = std::array::from_fn(|m| V32::from_slice(&MOVE_PERMS[m].centre_src));
        let wing_dst = std::array::from_fn(|m| {
            let mut d = V32::ZERO;
            for t in 0..24 {
                d.0[MOVE_PERMS[m].wing_src[t] as usize] = t as u8;
            }
            d
        });
        P1Tables {
            staging,
            table,
            big: false,
            frames,
            centre_idx,
            wing_dst,
        }
    }

    pub fn table_len() -> usize {
        N_CLASSES_A * N_SPLIT
    }

    #[inline(always)]
    pub fn value(&self, s: &P1State) -> u8 {
        let (ud, rl) = s.masks();
        if self.big {
            return self.table.get(self.staging.big_index(ud, rl));
        }
        self.table.get(
            self.staging
                .fast_index(unsafe { *self.staging.info(ud) }, rl),
        )
    }

    /// Exact staging distance, by walking down the table.
    pub fn distance(&self, s: &P1State) -> u8 {
        let mut s = *s;
        let mut d = 0;
        loop {
            let v = self.value(&s);
            let next = (0..N_MOVES)
                .map(|m| s.apply(m, self))
                .find(|c| self.value(c) == (v + 2) % 3);
            match next {
                Some(c) => {
                    s = c;
                    d += 1;
                }
                None => return d,
            }
        }
    }

    /// Which frames accept this staged state as an endpoint (bit f), given the last move.
    #[inline]
    pub fn endpoint_frames(&self, s: &P1State, last: usize) -> u8 {
        if s.parity {
            return 0;
        }
        let mut out = 0;
        for (f, fr) in self.frames.iter().enumerate() {
            if last < N_MOVES && fr.p2_mask >> last & 1 == 1 {
                continue;
            }
            if fr.wings_ok(&s.pos) && fr.centres_ok(&s.centres) {
                out |= 1 << f;
            }
        }
        out
    }
}

/// Depth-first search for phase-1 endpoints of exactly `depth` moves below a node.
pub struct Search1<'a, F: FnMut(&[u8], u8) -> bool> {
    pub t: &'a P1Tables,
    pub path: Vec<u8>,
    pub nodes: u64,
    pub on_endpoint: F,
    pub stop: bool,
    pub abort: Option<&'a std::sync::atomic::AtomicBool>,
}

impl<'a, F: FnMut(&[u8], u8) -> bool> Search1<'a, F> {
    pub fn new(t: &'a P1Tables, on_endpoint: F) -> Self {
        Search1 {
            t,
            path: Vec::with_capacity(32),
            nodes: 0,
            on_endpoint,
            stop: false,
            abort: None,
        }
    }

    /// Search below `s` (exact staging distance `d`) for paths of exactly `left` more moves.
    pub fn dfs(&mut self, s: &P1State, d: u8, left: u8, last: usize) {
        if self.t.big {
            self.dfs_t::<true>(s, d, left, last)
        } else {
            self.dfs_t::<false>(s, d, left, last)
        }
    }

    /// [`Search1::dfs`], for the large (`BIG`) or the small table.
    fn dfs_t<const BIG: bool>(&mut self, s: &P1State, d: u8, left: u8, last: usize) {
        let t = self.t;
        if left == 0 {
            if d == 0 {
                let frames = t.endpoint_frames(s, last);
                if frames != 0 && !(self.on_endpoint)(&self.path, frames) {
                    self.stop = true;
                }
            }
            return;
        }
        let mut allowed = ALLOWED_AFTER[last];
        if left == 1 {
            // The last move must leave the wing permutation even, and the wings pairable in a
            // frame where it is not a phase-2 move: both are tested before the centres.
            allowed &= if s.parity {
                WIDE_QUARTER
            } else {
                !WIDE_QUARTER
            };
            while allowed != 0 {
                let m = allowed.trailing_zeros() as usize;
                allowed &= allowed - 1;
                self.nodes += 1;
                let pos = t.wing_dst[m].gather(&s.pos);
                let mut frames = 0u8;
                for (f, fr) in t.frames.iter().enumerate() {
                    if fr.p2_mask >> m & 1 == 0 && fr.wings_ok(&pos) {
                        frames |= 1 << f;
                    }
                }
                if frames == 0 {
                    continue;
                }
                let c = s.apply(m, t);
                if Mod3Table::decode(d, t.value(&c)) != 0 {
                    continue;
                }
                for (f, fr) in t.frames.iter().enumerate() {
                    if frames >> f & 1 == 1 && !fr.centres_ok(&c.centres) {
                        frames &= !(1 << f);
                    }
                }
                if frames != 0 {
                    self.path.push(m as u8);
                    let go = (self.on_endpoint)(&self.path, frames);
                    self.path.pop();
                    if !go {
                        self.stop = true;
                        return;
                    }
                }
            }
            return;
        }
        // Stage 1: the children's centre masks, and a prefetch of their table entry (large
        // table) or class info (small table); stage 2 (small table): table indices,
        // prefetched; stage 3: in move order, descend into the children that can still reach
        // the goal. Only the table indices are kept: the few children descended into are
        // rebuilt from `s`.
        let mut idx = [0u64; N_MOVES];
        let mut nk = 0;
        let mut a = allowed;
        if BIG {
            while a != 0 {
                let m = a.trailing_zeros() as usize;
                a &= a - 1;
                let (ud, rl) = s.child_masks(m, t);
                let i = t.staging.big_index(ud, rl);
                t.table.prefetch(i);
                idx[nk] = i;
                nk += 1;
            }
        } else {
            let mut infos: [*const u32; N_MOVES] = [std::ptr::null(); N_MOVES];
            let mut rls = [0u32; N_MOVES];
            while a != 0 {
                let m = a.trailing_zeros() as usize;
                a &= a - 1;
                let (ud, rl) = s.child_masks(m, t);
                let p = t.staging.info(ud);
                crate::table::prefetch(p);
                infos[nk] = p;
                rls[nk] = rl;
                nk += 1;
            }
            for k in 0..nk {
                let i = t.staging.fast_index(unsafe { *infos[k] }, rls[k]);
                t.table.prefetch(i);
                idx[k] = i;
            }
        }
        self.nodes += nk as u64;
        if self
            .abort
            .is_some_and(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        {
            self.stop = true;
            return;
        }
        // The exact distance of a child from its stored value.
        let dec = [0, 1, 2, 3].map(|v| Mod3Table::decode(d, v));
        // With the large table, each value is read just before it is needed, so that the first
        // subtrees overlap the later children's table reads; the small table is mostly
        // cached, and reading all the values first is faster.
        let mut dcs = [0u8; N_MOVES];
        if !BIG {
            for k in 0..nk {
                dcs[k] = dec[t.table.get(idx[k]) as usize];
            }
        }
        for k in 0..nk {
            let m = allowed.trailing_zeros() as usize;
            allowed &= allowed - 1;
            let dc = if BIG {
                dec[t.table.get(idx[k]) as usize]
            } else {
                dcs[k]
            };
            if dc < left {
                let c = s.apply(m, t);
                self.path.push(m as u8);
                self.dfs_t::<BIG>(&c, dc, left - 1, m);
                self.path.pop();
                if self.stop {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_index_is_symmetric() {
        let st = Staging::new();
        let syms = centre_syms();
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let c = Cube::random(&mut rng);
            let s = P1State::from_cube(&c);
            let (ud, rl) = s.masks();
            let i = st.index(ud, rl);
            let fb = !(ud | rl) & MASK24;
            for p in &syms {
                let t = perm_tables(p);
                assert_eq!(st.index(apply_mask(&t, ud), apply_mask(&t, rl)), i);
                assert_eq!(st.index(apply_mask(&t, ud), apply_mask(&t, fb)), i);
            }
        }
    }

    #[test]
    fn frames_have_twelve_patterns() {
        for f in 0..3 {
            let fr = Frame::new(f);
            assert_eq!(fr.p2_mask.count_ones(), 17);
            let s = P1State::from_cube(&Cube::SOLVED);
            assert!(fr.wings_ok(&s.pos));
            assert!(fr.centres_ok(&s.centres));
        }
    }
}
