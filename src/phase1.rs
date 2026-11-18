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

/// The bit counts of the four bytes of `x` (one byte each).
#[inline(always)]
pub(crate) fn byte_counts(x: u32) -> u32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        use std::arch::aarch64::*;
        vget_lane_u32::<0>(vreinterpret_u32_u8(vcnt_u8(vcreate_u8(x as u64))))
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        (0..4).fold(0, |a, k| a | (x >> (8 * k) & 255).count_ones() << (8 * k))
    }
}

/// Without BMI2: per mask byte, per value byte, the value's bits under the mask, packed.
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
fn pext8_table() -> Box<[[u8; 256]; 256]> {
    let mut t: Box<[[u8; 256]; 256]> = vec![[0u8; 256]; 256].try_into().unwrap();
    for (m, row) in t.iter_mut().enumerate() {
        for (x, e) in row.iter_mut().enumerate() {
            *e = pext(x as u32, m as u32) as u8;
        }
    }
    t
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
    pub(crate) a_info: crate::table::HugeBuf<u32>,
    pub(crate) class_rep: Vec<u32>,
    /// Non-trivial stabiliser symmetries of each class (empty for most).
    pub(crate) stab: Vec<Vec<u8>>,
    /// Mask permutation tables: sym_perm[s][byte][value].
    sym_perm: Vec<[[u32; 256]; 3]>,
    /// The symmetries as byte gathers (position i of the image is position sym_gather[s][i]
    /// of the source; positions 24.. fixed), padded to 256 so that any info byte indexes it.
    pub(crate) sym_gather: Box<[V32; 256]>,
    pub(crate) move_perm: Vec<[[u32; 256]; 3]>,
    /// Unlabelled rank of an 8-of-16 mask (x and its complement share a rank).
    pub(crate) unl: Box<[u16; 1 << 16]>,
    pub(crate) unl_inv: Vec<u16>,
    /// Byte-sliced colex rank of 8-of-24 masks, by byte and number of bits below it (sized
    /// so that no count of the two lower bytes needs a bounds check; only 0..=8 are used).
    rank_t: [[[u32; 256]; 17]; 3],
    /// Without BMI2: [`pext`] of a byte by lookup (see [`Staging::pext24`]).
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    pext8: Box<[[u8; 256]; 256]>,
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
        let mut rank_t = [[[0u32; 256]; 17]; 3];
        for k in 0..3 {
            for c in 0..17 {
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
        let mut unl: Box<[u16; 1 << 16]> = vec![u16::MAX; 1 << 16].try_into().unwrap();
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
        let mut sym_gather: Box<[V32; 256]> = vec![V32(std::array::from_fn(|i| i as u8)); 256]
            .try_into()
            .unwrap();
        for (g, p) in sym_gather.iter_mut().zip(&syms) {
            for j in 0..24 {
                g.0[p[j] as usize] = j as u8;
            }
        }
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
            sym_gather,
            move_perm,
            unl,
            unl_inv,
            rank_t,
            #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
            pext8: pext8_table(),
        }
    }

    /// [`pext`] of 24-bit masks: BMI2 on x86-64, else three byte lookups.
    #[inline(always)]
    pub(crate) fn pext24(&self, x: u32, m: u32) -> u32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
        {
            pext(x, m)
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
        {
            let t = &self.pext8;
            let p0 = t[(m & 255) as usize][(x & 255) as usize] as u32;
            let p1 = t[(m >> 8 & 255) as usize][(x >> 8 & 255) as usize] as u32;
            let p2 = t[(m >> 16 & 255) as usize][(x >> 16 & 255) as usize] as u32;
            let c = byte_counts(m);
            let c0 = c & 255;
            p0 | p1 << c0 | p2 << (c0 + (c >> 8 & 255))
        }
    }

    #[inline(always)]
    pub fn rank24(&self, m: u32) -> u32 {
        let b0 = m & 255;
        let b1 = m >> 8 & 255;
        let b2 = m >> 16;
        let c = byte_counts(m);
        let c0 = (c & 255) as usize;
        let c1 = c0 + (c >> 8 & 255) as usize;
        debug_assert!(c0 <= 8 && c1 <= 16);
        // SAFETY: a byte has at most 8 bits set (c0 <= 8, c1 <= 16).
        unsafe {
            self.rank_t[0][0][b0 as usize]
                + self.rank_t.get_unchecked(1).get_unchecked(c0)[b1 as usize]
                + self.rank_t.get_unchecked(2).get_unchecked(c1)[b2 as usize]
        }
    }

    /// Canonical table index of the centre classes given the U/D-colour and R/L-colour
    /// position masks (the minimum over the class's stabiliser).
    pub fn index(&self, ud: u32, rl: u32) -> u64 {
        let info = self.a_info[self.rank24(ud) as usize];
        let class = (info >> 8) as usize;
        let s = (info & 255) as usize;
        let rem = !self.class_rep[class] & MASK24;
        let b = apply_mask(&self.sym_perm[s], rl);
        let mut u = self.unl[pext(b, rem) as u16 as usize];
        for &t in &self.stab[class] {
            let bt = apply_mask(&self.sym_perm[t as usize], b);
            u = u.min(self.unl[pext(bt, rem) as u16 as usize]);
        }
        (class * N_SPLIT + u as usize) as u64
    }

    /// The class info (class << 8 | symmetry to the representative) of a U/D mask; it is the
    /// first, separately prefetchable, step of [`Staging::fast_index`].
    #[inline(always)]
    pub fn info(&self, ud: u32) -> *const u32 {
        unsafe { self.a_info.as_ptr().add(self.rank24(ud) as usize) }
    }

    /// [`Staging::info`] given the rank of the U/D mask.
    #[inline(always)]
    pub fn info_of_rank(&self, rank: u64) -> *const u32 {
        unsafe { self.a_info.as_ptr().add(rank as usize) }
    }

    /// A table index of the centre classes, without minimising over the stabiliser: the
    /// table holds the same value at every stabiliser variant.
    #[inline(always)]
    pub fn fast_index(&self, info: u32, rl: u32) -> u64 {
        let class = (info >> 8) as usize;
        let s = (info & 255) as usize;
        let b = apply_mask(&self.sym_perm[s], rl);
        let rem = !self.class_rep[class] & MASK24;
        (class * N_SPLIT + self.unl[self.pext24(b, rem) as u16 as usize] as usize) as u64
    }

    /// [`Staging::fast_index`] from the flags of the centres (see [`P1State::centre_flags`]).
    #[inline(always)]
    pub(crate) fn fast_index_flags(&self, info: u32, flags: &V32) -> u64 {
        let class = (info >> 8) as usize;
        let h = flags.gather(&self.sym_gather[(info & 255) as usize]);
        let b = crate::simd::movemask(&h);
        let rem = crate::simd::movemask(&crate::simd::add(&h, &h));
        (class * N_SPLIT + self.unl[self.pext24(b, rem) as u16 as usize] as usize) as u64
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
        self.big_index_n(ud, !ud & MASK24, rl)
    }

    /// [`Staging::big_index`], given also the complement `nud` of the U/D mask.
    #[inline(always)]
    pub(crate) fn big_index_n(&self, ud: u32, nud: u32, rl: u32) -> u64 {
        self.rank24(ud) as u64 * N_SPLIT as u64
            + self.unl[self.pext24(rl, nud) as u16 as usize] as u64
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
        self.after_pair(moves, d, t, ALL_PAIRS)
    }

    /// [`P1State::after`] with the distance of the search over `pair`.
    pub fn after_pair(&self, moves: &[u8], d: u8, t: &P1Tables, pair: usize) -> (P1State, u8) {
        moves.iter().fold((*self, d), |(s, d), &m| {
            let c = s.apply(m as usize, t);
            (c, Mod3Table::decode(d, t.value_pair(&c, pair)))
        })
    }

    /// The U/D-colour and R/L-colour position masks of the centres.
    #[inline(always)]
    pub fn masks(&self) -> (u32, u32) {
        centre_masks(self.centres)
    }

    /// Per centre position, bit 7 set for an R/L colour and bit 6 for any colour but U/D
    /// (zero at positions 24.., which every move leaves in place).
    #[inline(always)]
    pub(crate) fn centre_flags(&self) -> V32 {
        const LUT: V32 = {
            let mut t = [0u8; 32];
            let mut i = 0;
            while i < 32 {
                t[i] = match i & 15 {
                    2 | 3 => 0xc0,
                    4 | 5 => 0x40,
                    _ => 0,
                };
                i += 1;
            }
            V32(t)
        };
        const LOW24: V32 = {
            let mut t = [0u8; 32];
            let mut i = 0;
            while i < 24 {
                t[i] = 0xff;
                i += 1;
            }
            V32(t)
        };
        crate::simd::and(&crate::simd::shuf(&LUT, &self.centres), &LOW24)
    }
}

/// The [`P1State::centre_flags`] of the child after move `m`, from the parent's.
///
/// # Safety
/// `m < N_MOVES`.
#[inline(always)]
pub(crate) unsafe fn child_flags(flags: &V32, m: usize, t: &P1Tables) -> V32 {
    debug_assert!(m < N_MOVES);
    flags.gather(unsafe { t.centre_idx.get_unchecked(m) })
}

/// The U/D-colour, other-colour and R/L-colour position masks of centres, from their
/// [`P1State::centre_flags`].
#[inline(always)]
pub(crate) fn flag_masks(g: &V32) -> (u32, u32, u32) {
    let rl = crate::simd::movemask(g);
    let nud = crate::simd::movemask(&crate::simd::add(g, g));
    debug_assert!((rl | nud) >> 24 == 0);
    (!nud & MASK24, nud, rl)
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
    /// Per move m, per slot j: bit 5 + f set if the slot that m takes j to is low in frame f
    /// (see [`P1Tables::wing_frames`]; zero at j >= 24).
    low_after: [V32; N_MOVES],
    /// Per piece (byte), bit 5 + f set if its home slot is high in frame f.
    high_bytes: [u64; 3],
    /// Per move, the frames (bit f) in which it is a phase-2 move.
    p2_frames: [u8; N_MOVES],
    /// The pattern table (searches per colour pair, see [`crate::p1pat`]).
    pub pat: Option<crate::p1pat::PatTable>,
    /// The move-mask table (with the large staging table, see [`crate::p1mask`]).
    pub mask: Option<crate::p1mask::MaskTable>,
}

/// [`Search1::pair`] for the search over all pairs, pruned by the staging table only.
pub const ALL_PAIRS: usize = 3;

impl P1Tables {
    pub fn with_table(staging: Staging, table: Mod3Table) -> P1Tables {
        let frames = [Frame::new(0), Frame::new(1), Frame::new(2)];
        // Positions 24.. map to themselves (see P1State::centre_flags).
        let centre_idx = std::array::from_fn(|m| {
            let mut v = V32(std::array::from_fn(|i| i as u8));
            v.0[..24].copy_from_slice(&MOVE_PERMS[m].centre_src);
            v
        });
        let wing_dst = std::array::from_fn(|m| {
            let mut d = V32::ZERO;
            for t in 0..24 {
                d.0[MOVE_PERMS[m].wing_src[t] as usize] = t as u8;
            }
            d
        });
        let mut low = [0u8; 24];
        let mut high_bytes = [0u64; 3];
        for (f, fr) in frames.iter().enumerate() {
            for &sl in &fr.low_slots {
                low[sl as usize] |= 1 << (5 + f);
            }
            for i in 0..24 {
                if fr.high_pieces >> i & 1 == 1 {
                    high_bytes[i / 8] |= 1 << (5 + f + 8 * (i % 8));
                }
            }
        }
        let low_after = std::array::from_fn(|m| {
            let mut v = V32::ZERO;
            for j in 0..24 {
                v.0[j] = low[wing_dst[m].0[j] as usize];
            }
            v
        });
        let p2_frames = std::array::from_fn(|m| {
            (0..3)
                .filter(|&f| frames[f].p2_mask >> m & 1 == 1)
                .fold(0, |a, f| a | 1 << f)
        });
        P1Tables {
            staging,
            table,
            big: false,
            frames,
            centre_idx,
            wing_dst,
            low_after,
            high_bytes,
            p2_frames,
            pat: None,
            mask: None,
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

    /// The stored value of a state for the search over colour pair `pair` (its pattern
    /// table), or over all pairs ([`ALL_PAIRS`]: the staging table).
    #[inline(always)]
    pub fn value_pair(&self, s: &P1State, pair: usize) -> u8 {
        if pair == ALL_PAIRS {
            self.value(s)
        } else {
            self.pat.as_ref().unwrap().value(&self.staging, s, pair)
        }
    }

    /// Exact staging distance, by walking down the table.
    pub fn distance(&self, s: &P1State) -> u8 {
        self.distance_pair(s, ALL_PAIRS)
    }

    /// Exact distance for the search over `pair` (see [`P1Tables::value_pair`]).
    pub fn distance_pair(&self, s: &P1State, pair: usize) -> u8 {
        let mut s = *s;
        let mut d = 0;
        loop {
            let v = self.value_pair(&s, pair);
            let next = (0..N_MOVES)
                .map(|m| s.apply(m, self))
                .find(|c| self.value_pair(c, pair) == (v + 2) % 3);
            match next {
                Some(c) => {
                    s = c;
                    d += 1;
                }
                None => return d,
            }
        }
    }

    /// The frames (bit f) whose special axis holds colour pair `pair` (staged centres), or all
    /// frames for [`ALL_PAIRS`].
    #[inline]
    pub fn pair_frames(&self, centres: &V32, pair: usize) -> u8 {
        if pair == ALL_PAIRS {
            return 7;
        }
        let mut out = 0;
        for (f, fr) in self.frames.iter().enumerate() {
            if (centres.0[fr.rl_pos[0] as usize] >> 1) as usize == pair {
                out |= 1 << f;
            }
        }
        out
    }

    /// The frames (bit f) in which the wings are pairable ([`Frame::wings_ok`]) after move `m`
    /// from wing positions `pos`: all three tests on one gather.
    #[inline(always)]
    pub fn wing_frames(&self, pos: &V32, m: usize) -> u32 {
        debug_assert!(m < N_MOVES);
        let x = unsafe { self.low_after.get_unchecked(m) }.gather(pos);
        let w = |k: usize| u64::from_le_bytes(x.0[8 * k..8 * k + 8].try_into().unwrap());
        let (w0, w1, w2) = (w(0), w(1), w(2));
        // Pieces 2k and 2k + 1 (a dedge): exactly one low (even bytes of w ^ w >> 8).
        let t = (w0 ^ w0 >> 8) & (w1 ^ w1 >> 8) & (w2 ^ w2 >> 8);
        let t = t & t >> 16;
        let t = t & t >> 32;
        // An even number of low wings among the high pieces.
        let mut p =
            (w0 & self.high_bytes[0]) ^ (w1 & self.high_bytes[1]) ^ (w2 & self.high_bytes[2]);
        p ^= p >> 32;
        p ^= p >> 16;
        p ^= p >> 8;
        ((t & !p) >> 5) as u32 & 7
    }

    /// Which frames accept this staged state as an endpoint (bit f), given the trailing run
    /// of same-axis moves of its path ([`trailing_run`]).
    #[inline]
    pub fn endpoint_frames(&self, s: &P1State, run: u32) -> u8 {
        if s.parity {
            return 0;
        }
        let mut out = 0;
        for (f, fr) in self.frames.iter().enumerate() {
            if fr.p2_mask & run != 0 {
                continue;
            }
            if fr.wings_ok(&s.pos) && fr.centres_ok(&s.centres) {
                out |= 1 << f;
            }
        }
        out
    }
}

/// The moves (a bit per move index) of the trailing run of same-axis moves of `path`.
#[inline]
pub fn trailing_run(path: &[u8]) -> u32 {
    let mut run = 0;
    if let Some(&last) = path.last() {
        let axis = (last / 3) % 3;
        for &m in path.iter().rev().take_while(|&&m| (m / 3) % 3 == axis) {
            run |= 1 << m;
        }
    }
    run
}

/// Depth-first search for phase-1 endpoints of exactly `depth` moves below a node.
pub struct Search1<'a, F: FnMut(&[u8], u8) -> bool> {
    pub t: &'a P1Tables,
    pub path: Vec<u8>,
    pub nodes: u64,
    /// Table entries read (each a likely cache miss): children's distances, move masks.
    pub reads: u64,
    pub on_endpoint: F,
    pub stop: bool,
    pub abort: Option<&'a std::sync::atomic::AtomicBool>,
    /// The colour pair whose endpoints (those whose special axis holds it) are searched for,
    /// pruned by its pattern table; [`ALL_PAIRS`]: all endpoints, pruned by the staging table.
    pub pair: usize,
    /// With the move-mask table: nodes at staging distance = moves left (>= 2), expanded in
    /// batches (see [`Search1::run_batch`]).
    batch: Vec<BNode>,
    batch_next: Vec<BNode>,
    /// The paths of the batch nodes, as a tree: per node (by [`BNode::id`]), its parent's id
    /// and its last move, or for a node queued by the depth-first search, [`ROOT`] | the
    /// offset in `batch_roots` of its path (its length, then the moves).
    batch_tree: Vec<(u32, u8)>,
    batch_roots: Vec<u8>,
    batch_path: Vec<u8>,
}

/// See [`Search1::batch_tree`].
const ROOT: u32 = 1 << 31;

/// A node of a batch (moves left = staging distance): its state, mask-table index, and its
/// path's entry in [`Search1::batch_tree`].
#[derive(Clone, Copy)]
struct BNode {
    centres: V32,
    pos: V32,
    /// The mask table's first index (see `MaskTable::pre_index`; for the reduced and pair
    /// tables, their index, once resolved at the start of the node's round, with `sym`).
    idx: u64,
    id: u32,
    /// The moves of its path's trailing run of same-axis moves (see [`trailing_run`]).
    run: u32,
    parity: bool,
    sym: u8,
    left: u8,
    last: u8,
}

impl BNode {
    #[inline(always)]
    fn state(&self) -> P1State {
        P1State {
            centres: self.centres,
            pos: self.pos,
            parity: self.parity,
        }
    }
}

/// The trailing run of same-axis moves after move `m`, given the run `run` before it and the
/// move `last` before it.
#[inline(always)]
fn run_after(run: u32, last: usize, m: usize) -> u32 {
    if last < N_MOVES && (m / 3) % 3 == (last / 3) % 3 {
        run | 1 << m
    } else {
        1 << m
    }
}

/// The table guiding a search ([`Search1::dfs_t`]'s `K`).
const K_SMALL: u8 = 0;
const K_BIG: u8 = 1;
const K_PAT: u8 = 2;
const K_PAT_BIG: u8 = 3;
/// The large staging table, and the move-mask table for nodes at distance = moves left.
const K_MASK: u8 = 4;

impl<'a, F: FnMut(&[u8], u8) -> bool> Search1<'a, F> {
    pub fn new(t: &'a P1Tables, on_endpoint: F) -> Self {
        Search1 {
            t,
            path: Vec::with_capacity(32),
            nodes: 0,
            reads: 0,
            on_endpoint,
            stop: false,
            abort: None,
            pair: ALL_PAIRS,
            batch: vec![],
            batch_next: vec![],
            batch_tree: vec![],
            batch_roots: vec![],
            batch_path: Vec::with_capacity(48),
        }
    }

    /// Queue a node at staging distance `left` = moves left (>= 2) for a batch: the state `s`
    /// after the search's path and `m` (if any), its mask table's first index `idx`.
    #[inline(always)]
    fn push_batch(&mut self, s: P1State, idx: u64, left: u8, last: usize, m: Option<u8>) {
        let off = self.batch_roots.len();
        self.batch_roots.push(0);
        self.batch_roots.extend_from_slice(&self.path);
        if let Some(m) = m {
            self.batch_roots.push(m);
        }
        let n = self.batch_roots.len() - off - 1;
        self.batch_roots[off] = n as u8;
        let run = trailing_run(&self.batch_roots[off + 1..]);
        let id = self.batch_tree.len() as u32;
        self.batch_tree.push((ROOT | off as u32, 0));
        self.t.mask.as_ref().unwrap().prefetch(self.t, idx);
        self.batch.push(BNode {
            centres: s.centres,
            pos: s.pos,
            idx,
            id,
            run,
            parity: s.parity,
            sym: 0,
            left,
            last: last as u8,
        });
    }

    /// Queue the child `c` (after move `m`, first index `idx`) of batch node `nd`.
    #[inline(always)]
    fn push_child(&mut self, nd: &BNode, c: &P1State, idx: u64, m: usize) {
        let id = self.batch_tree.len() as u32;
        self.batch_tree.push((nd.id, m as u8));
        self.t.mask.as_ref().unwrap().prefetch(self.t, idx);
        self.batch.push(BNode {
            centres: c.centres,
            pos: c.pos,
            idx,
            id,
            run: run_after(nd.run, nd.last as usize, m),
            parity: c.parity,
            sym: 0,
            left: nd.left - 1,
            last: m as u8,
        });
    }

    /// Set `self.path` to the path of batch node `id` (see [`Search1::batch_tree`]).
    #[cold]
    fn set_batch_path(&mut self, id: u32) {
        let mut tail = vec![];
        let mut e = self.batch_tree[id as usize];
        while e.0 & ROOT == 0 {
            tail.push(e.1);
            e = self.batch_tree[e.0 as usize];
        }
        let off = (e.0 & !ROOT) as usize;
        let n = self.batch_roots[off] as usize;
        self.path.clear();
        self.path
            .extend_from_slice(&self.batch_roots[off + 1..off + 1 + n]);
        self.path.extend(tail.iter().rev());
    }

    /// Search the subtrees of the queued nodes: in rounds, each node's children are the
    /// distance-decreasing moves of its mask (read a round after its prefetch); children
    /// with one move left get the last-move tests, the others are queued for the next round.
    fn run_batch(&mut self) {
        // The batch nodes' paths replace the search's path meanwhile.
        std::mem::swap(&mut self.path, &mut self.batch_path);
        self.run_batch_rounds();
        std::mem::swap(&mut self.path, &mut self.batch_path);
        self.batch_tree.clear();
        self.batch_roots.clear();
    }

    fn run_batch_rounds(&mut self) {
        let t = self.t;
        let mt = t.mask.as_ref().unwrap();
        let reduced = mt.kind == crate::p1mask::MaskKind::Reduced;
        while !self.batch.is_empty() {
            let mut cur = std::mem::take(&mut self.batch);
            std::mem::swap(&mut self.batch, &mut self.batch_next);
            self.batch.clear();
            if self
                .abort
                .is_some_and(|a| a.load(std::sync::atomic::Ordering::Relaxed))
            {
                self.stop = true;
            }
            if self.stop {
                self.batch.clear();
                return;
            }
            if matches!(
                mt.kind,
                crate::p1mask::MaskKind::Reduced | crate::p1mask::MaskKind::Pair
            ) {
                // The reduced table's index needs the class info (prefetched at the push).
                for nd in cur.iter_mut() {
                    (nd.idx, nd.sym) = mt.resolve(t, nd.idx, &nd.state(), self.pair);
                }
            }
            for nd in &cur {
                let s = nd.state();
                let mut a =
                    mt.get(t, nd.idx, nd.sym, &s, self.pair) & ALLOWED_AFTER[nd.last as usize];
                self.nodes += a.count_ones() as u64;
                self.reads += 1;
                if nd.left == 1 {
                    self.leaf(
                        &s,
                        1,
                        nd.last as usize,
                        Some(a),
                        nd.run,
                        Some((nd.id, None)),
                    );
                    if self.stop {
                        self.batch.clear();
                        return;
                    }
                } else if nd.left == 2 && !mt.batch_leaves {
                    while a != 0 {
                        let m = a.trailing_zeros() as usize;
                        a &= a - 1;
                        let c = s.apply(m, t);
                        let run = run_after(nd.run, nd.last as usize, m);
                        self.leaf(&c, 1, m, None, run, Some((nd.id, Some(m as u8))));
                        if self.stop {
                            self.batch.clear();
                            return;
                        }
                    }
                } else if mt.kind == crate::p1mask::MaskKind::Pair {
                    // The children's first index: the rank of the pair's positions.
                    let flags = t.pat.as_ref().unwrap().flags(&s.centres, self.pair);
                    while a != 0 {
                        let m = a.trailing_zeros() as usize;
                        a &= a - 1;
                        let g = unsafe { child_flags(&flags, m, t) };
                        let i = t.staging.rank24(crate::simd::movemask(&g)) as u64;
                        let c = s.apply(m, t);
                        self.push_child(nd, &c, i, m);
                    }
                } else {
                    let flags = s.centre_flags();
                    while a != 0 {
                        let m = a.trailing_zeros() as usize;
                        a &= a - 1;
                        let g = unsafe { child_flags(&flags, m, t) };
                        // The reduced table needs only the rank of the U/D positions.
                        let i = if reduced {
                            let nud = crate::simd::movemask(&crate::simd::add(&g, &g));
                            t.staging.rank24(!nud & MASK24) as u64
                        } else {
                            let (ud, nud, rl) = flag_masks(&g);
                            t.staging.big_index_n(ud, nud, rl)
                        };
                        let c = s.apply(m, t);
                        self.push_child(nd, &c, i, m);
                    }
                }
            }
            cur.clear();
            self.batch_next = cur;
        }
    }

    /// The fewest moves left of a node queued for a batch (1 if the batches also read the
    /// masks of nodes one move from the end, to test only their staging last moves).
    #[inline(always)]
    fn batch_min_left(&self) -> u8 {
        if self.t.mask.as_ref().is_some_and(|m| m.batch_leaves) {
            1
        } else {
            2
        }
    }

    /// The stored value of a state in the table guiding this search.
    #[inline(always)]
    pub fn value(&self, s: &P1State) -> u8 {
        self.t.value_pair(s, self.pair)
    }

    /// Search below `s` (exact distance `d` in the table guiding this search: staging, or
    /// the pair's pattern table) for paths of exactly `left` more moves.
    pub fn dfs(&mut self, s: &P1State, d: u8, left: u8, last: usize) {
        if self.pair != ALL_PAIRS {
            if self.t.pat.as_ref().unwrap().big {
                self.dfs_t::<K_PAT_BIG>(s, d, left, last)
            } else {
                self.dfs_t::<K_PAT>(s, d, left, last)
            }
            if self.t.mask.is_some() && !self.stop {
                self.run_batch();
            }
        } else if self.t.big
            && self
                .t
                .mask
                .as_ref()
                .is_some_and(|m| m.kind != crate::p1mask::MaskKind::Pair)
        {
            self.dfs_t::<K_MASK>(s, d, left, last);
            if !self.stop {
                self.run_batch();
            }
        } else if self.t.big {
            self.dfs_t::<K_BIG>(s, d, left, last)
        } else {
            self.dfs_t::<K_SMALL>(s, d, left, last)
        }
    }

    /// The last move below `s` (exact distance `d`, one move left): endpoint tests.
    /// `staging`: the moves that stage the centres, if known (from a move mask, `d` = 1).
    /// `run`: the trailing run of same-axis moves of the path to `s` ([`trailing_run`]).
    /// `batch`: the path to `s` is that of batch node `id` and move `m` (if any), not
    /// `self.path`.
    fn leaf(
        &mut self,
        s: &P1State,
        d: u8,
        last: usize,
        staging: Option<u32>,
        run: u32,
        batch: Option<(u32, Option<u8>)>,
    ) {
        // The last move must leave the wing permutation even, and the wings pairable in a
        // frame where the path's trailing run of same-axis moves has no phase-2 move (else
        // the shorter endpoint without them, with them opening phase 2, covers it): both
        // are tested before the centres.
        let mut allowed = ALLOWED_AFTER[last] & staging.unwrap_or(u32::MAX);
        allowed &= if s.parity {
            WIDE_QUARTER
        } else {
            !WIDE_QUARTER
        };
        // The frames in which the run before the last move has a phase-2 move (the last move
        // continues the run if on the same axis).
        let mut run_frames = 0;
        let mut r = run;
        while r != 0 {
            run_frames |= self.t.p2_frames[r.trailing_zeros() as usize];
            r &= r - 1;
        }
        while allowed != 0 {
            let m = allowed.trailing_zeros() as usize;
            allowed &= allowed - 1;
            self.nodes += 1;
            let mut excl = unsafe { *self.t.p2_frames.get_unchecked(m) };
            if last < N_MOVES && (m / 3) % 3 == (last / 3) % 3 {
                excl |= run_frames;
            }
            let mut frames = (self.t.wing_frames(&s.pos, m) & !excl as u32) as u8;
            if frames == 0 {
                continue;
            }
            let c = s.apply(m, self.t);
            if staging.is_none() {
                self.reads += 1;
                if Mod3Table::decode(d, self.value(&c)) != 0 {
                    continue;
                }
            }
            frames &= self.t.pair_frames(&c.centres, self.pair);
            for (f, fr) in self.t.frames.iter().enumerate() {
                if frames >> f & 1 == 1 && !fr.centres_ok(&c.centres) {
                    frames &= !(1 << f);
                }
            }
            if frames != 0 {
                if let Some((id, mv)) = batch {
                    self.set_batch_path(id);
                    self.path.extend(mv);
                }
                self.path.push(m as u8);
                let go = (self.on_endpoint)(&self.path, frames);
                self.path.pop();
                if !go {
                    self.stop = true;
                    return;
                }
            }
        }
    }

    /// [`Search1::dfs`], guided by the small or large staging table, by the reduced or
    /// large pattern table, or by the large staging table and the move masks (`K`).
    fn dfs_t<const K: u8>(&mut self, s: &P1State, d: u8, left: u8, last: usize) {
        if left == 0 {
            if d == 0 {
                let frames = self.t.endpoint_frames(s, trailing_run(&self.path))
                    & self.t.pair_frames(&s.centres, self.pair);
                if frames != 0 && !(self.on_endpoint)(&self.path, frames) {
                    self.stop = true;
                }
            }
            return;
        }
        if left == 1 {
            self.leaf(s, d, last, None, trailing_run(&self.path), None);
            return;
        }
        // Batches: with the move masks, or (counts only) per-pair searches with the virtual
        // move-mask table, which then gives the pattern-distance-decreasing moves.
        let batched = K == K_MASK || (K >= K_PAT && K <= K_PAT_BIG && self.t.mask.is_some());
        if batched && d == left {
            let i = self
                .t
                .mask
                .as_ref()
                .unwrap()
                .pre_index(self.t, s, self.pair);
            self.push_batch(*s, i, left, last, None);
            return;
        }
        let mut allowed = ALLOWED_AFTER[last];
        let t = self.t;
        // Stage 1: the children's centre masks, and a prefetch of their table entry (large
        // tables) or class info (reduced tables); stage 2 (reduced tables): table indices,
        // prefetched; stage 3: in move order, descend into the children that can still reach
        // the goal. Only the table indices are kept: the few children descended into are
        // rebuilt from `s`.
        let mut idx = [0u64; N_MOVES];
        let mut nk = 0;
        let mut a = allowed;
        // SAFETY (the unchecked indexing below): the allowed moves are < N_MOVES, and so is
        // the number of children before the last.
        debug_assert!(allowed >> N_MOVES == 0);
        if K == K_BIG || K == K_MASK {
            let flags = s.centre_flags();
            while a != 0 {
                let m = a.trailing_zeros() as usize;
                a &= a - 1;
                let (ud, nud, rl) = flag_masks(&unsafe { child_flags(&flags, m, t) });
                let i = t.staging.big_index_n(ud, nud, rl);
                t.table.prefetch(i);
                unsafe { *idx.get_unchecked_mut(nk) = i };
                nk += 1;
            }
        } else if K == K_PAT_BIG {
            let pt = t.pat.as_ref().unwrap();
            let flags = pt.flags(&s.centres, self.pair);
            while a != 0 {
                let m = a.trailing_zeros() as usize;
                a &= a - 1;
                let i = pt.big_index(&t.staging, &unsafe { child_flags(&flags, m, t) });
                pt.table.prefetch(i);
                unsafe { *idx.get_unchecked_mut(nk) = i };
                nk += 1;
            }
        } else {
            let pt = t.pat.as_ref();
            let flags = if K == K_PAT {
                pt.unwrap().flags(&s.centres, self.pair)
            } else {
                s.centre_flags()
            };
            let mut infos: [*const u32; N_MOVES] = [std::ptr::null(); N_MOVES];
            let mut gs = [V32::ZERO; N_MOVES];
            while a != 0 {
                let m = a.trailing_zeros() as usize;
                a &= a - 1;
                let g = unsafe { child_flags(&flags, m, t) };
                let p = if K == K_PAT {
                    // The pair's positions: bit 7 of the flags.
                    t.staging.info(crate::simd::movemask(&g))
                } else {
                    t.staging.info(flag_masks(&g).0)
                };
                crate::table::prefetch(p);
                unsafe {
                    *infos.get_unchecked_mut(nk) = p;
                    *gs.get_unchecked_mut(nk) = g;
                }
                nk += 1;
            }
            for ((i, &p), g) in idx.iter_mut().zip(&infos[..nk]).zip(&gs) {
                if K == K_PAT {
                    let pt = pt.unwrap();
                    *i = pt.red_index(&t.staging, unsafe { *p }, g);
                    pt.table.prefetch(*i);
                } else {
                    *i = t.staging.fast_index_flags(unsafe { *p }, g);
                    t.table.prefetch(*i);
                }
            }
        }
        let idx = &idx[..nk];
        self.nodes += nk as u64;
        self.reads += nk as u64;
        if self
            .abort
            .is_some_and(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        {
            self.stop = true;
            return;
        }
        let table = if K == K_PAT || K == K_PAT_BIG {
            &t.pat.as_ref().unwrap().table
        } else {
            &t.table
        };
        // The exact distance of a child from its stored value.
        let dec = [0, 1, 2, 3].map(|v| Mod3Table::decode(d, v));
        // With the large (or pattern) tables, each value is read just before it is needed, so
        // that the first subtrees overlap the later children's table reads; the small staging
        // table is mostly cached, and reading all the values first is faster.
        let mut dcs = [0u8; N_MOVES];
        if K == K_SMALL {
            for (dc, &i) in dcs.iter_mut().zip(idx) {
                *dc = dec[table.get(i) as usize];
            }
        }
        let dcs = &dcs[..nk];
        for k in 0..nk {
            let m = allowed.trailing_zeros() as usize;
            allowed &= allowed - 1;
            let dc = if K != K_SMALL {
                dec[table.get(idx[k]) as usize]
            } else {
                dcs[k]
            };
            if dc < left {
                let c = s.apply(m, t);
                if batched && dc == left - 1 && dc >= self.batch_min_left() {
                    let i = if K == K_MASK {
                        t.mask.as_ref().unwrap().pre_index_of_big(idx[k])
                    } else {
                        t.mask.as_ref().unwrap().pre_index(t, &c, self.pair)
                    };
                    self.push_batch(c, i, dc, m, Some(m as u8));
                    if self.batch.len() >= t.mask.as_ref().unwrap().batch {
                        self.run_batch();
                    }
                } else {
                    self.path.push(m as u8);
                    self.dfs_t::<K>(&c, dc, left - 1, m);
                    self.path.pop();
                }
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
    fn pext24_matches_pext() {
        use rand::Rng;
        let st = Staging::new();
        let mut rng = rand::thread_rng();
        for _ in 0..100000 {
            let (x, m) = (rng.r#gen::<u32>() & MASK24, rng.r#gen::<u32>() & MASK24);
            assert_eq!(st.pext24(x, m), pext(x, m));
            assert_eq!(
                byte_counts(x),
                (0..4).fold(0, |a, k| a | (x >> (8 * k) & 255).count_ones() << (8 * k))
            );
        }
    }

    #[test]
    fn child_masks_match_apply() {
        let t = P1Tables::with_table(Staging::new(), Mod3Table::from_depths(&[0]));
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let mut s = P1State::from_cube(&Cube::random(&mut rng));
            for step in 0..20 {
                let flags = s.centre_flags();
                for m in 0..N_MOVES {
                    let c = s.apply(m, &t);
                    let (ud, rl) = c.masks();
                    let g = unsafe { child_flags(&flags, m, &t) };
                    assert_eq!(flag_masks(&g), (ud, !ud & MASK24, rl));
                    assert_eq!(
                        t.staging.big_index_n(ud, !ud & MASK24, rl),
                        t.staging.big_index(ud, rl)
                    );
                    let info = unsafe { *t.staging.info(ud) };
                    assert_eq!(
                        t.staging.fast_index_flags(info, &g),
                        t.staging.fast_index(info, rl)
                    );
                }
                s = s.apply((step * 7 + 3) % N_MOVES, &t);
            }
        }
    }

    /// All (endpoint, frame) pairs of length `len` below `root` (sorted), by the search over
    /// all pairs or by the three per-pair searches.
    fn endpoint_set(t: &P1Tables, root: &P1State, len: u8, pairs: bool) -> Vec<(Vec<u8>, u8)> {
        let mut found = vec![];
        let which: Vec<usize> = if pairs {
            vec![0, 1, 2]
        } else {
            vec![ALL_PAIRS]
        };
        for pair in which {
            let d = t.distance_pair(root, pair);
            if d > len {
                continue;
            }
            let mut s = Search1::new(t, |p: &[u8], f| {
                for g in 0..3 {
                    if f >> g & 1 == 1 {
                        found.push((p.to_vec(), g));
                    }
                }
                true
            });
            s.pair = pair;
            s.dfs(root, d, len, N_MOVES);
        }
        found.sort();
        found
    }

    /// The batched search with move masks, and the per-pair searches with the pattern table,
    /// find exactly the plain search's (endpoint, frame) pairs (needs the large staging table,
    /// and for the pairs the reduced pattern table, in ./tables; skipped without them).
    #[test]
    fn masked_and_pair_searches_find_the_same_endpoints() {
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-staging-big.tab").exists() {
            return;
        }
        let mut t = crate::solver::load_p1(dir, true, 8, false);
        let with_pat = dir.join("p1-pat.tab").exists();
        if with_pat {
            crate::solver::load_pat(&mut t, dir, false, 8, false);
        }
        let mut rng = rand::thread_rng();
        for _ in 0..3 {
            let root = P1State::from_cube(&Cube::random(&mut rng));
            let d0 = t.distance(&root);
            for len in d0..d0 + 2 {
                t.mask = None;
                let want = endpoint_set(&t, &root, len, false);
                t.mask = Some(crate::p1mask::MaskTable::virtual_table());
                assert_eq!(endpoint_set(&t, &root, len, false), want);
                t.mask = None;
                if with_pat {
                    assert_eq!(endpoint_set(&t, &root, len, true), want);
                }
            }
        }
    }

    /// [`P1Tables::wing_frames`] agrees with the per-frame tests.
    #[test]
    fn wing_frames_match_wings_ok() {
        let t = P1Tables::with_table(Staging::new(), Mod3Table::from_depths(&[0]));
        let mut rng = rand::thread_rng();
        let mut seen = [0u32; 8];
        for k in 0..20000 {
            let mut s = P1State::from_cube(&Cube::random(&mut rng));
            if k % 2 == 0 {
                // Near-pairable states too: a few moves from solved.
                s = P1State::from_cube(&Cube::SOLVED);
                for j in 0..k % 7 {
                    s = s.apply((k * 13 + j * 5) % N_MOVES, &t);
                }
            }
            for m in 0..N_MOVES {
                let pos = t.wing_dst[m].gather(&s.pos);
                let want = (0..3)
                    .filter(|&f| t.frames[f].wings_ok(&pos))
                    .fold(0, |a, f| a | 1 << f);
                assert_eq!(t.wing_frames(&s.pos, m), want);
                seen[want as usize] += 1;
            }
        }
        assert!(seen[0] > 0 && seen[7] > 0 && seen[1..7].iter().any(|&n| n > 0));
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

    /// The allowed special-axis patterns are the same set in every frame up to the 48
    /// symmetries: the image of an allowed pattern of frame f under any symmetry is an allowed
    /// pattern of the frame whose special axis the symmetry maps f's onto.
    #[test]
    fn patterns_are_symmetric() {
        let frames = [Frame::new(0), Frame::new(1), Frame::new(2)];
        let syms = centre_syms();
        let set = |fr: &Frame| -> u32 { fr.rl_pos.iter().map(|&p| 1u32 << p).sum() };
        let mut bad = 0;
        for (f, fr) in frames.iter().enumerate() {
            for p in 0u32..256 {
                if fr.rl_ok[p as usize >> 6] >> (p & 63) & 1 == 0 {
                    continue;
                }
                for (si, s) in syms.iter().enumerate() {
                    let img_set: u32 = fr.rl_pos.iter().map(|&q| 1u32 << s[q as usize]).sum();
                    let g = (0..3).find(|&g| set(&frames[g]) == img_set).unwrap();
                    let mut img = 0u32;
                    for i in 0..8 {
                        if p >> i & 1 == 1 {
                            img |= 1 << s[fr.rl_pos[i] as usize];
                        }
                    }
                    let mut q = 0u32;
                    for (i, &r) in frames[g].rl_pos.iter().enumerate() {
                        if img >> r & 1 == 1 {
                            q |= 1 << i;
                        }
                    }
                    if frames[g].rl_ok[q as usize >> 6] >> (q & 63) & 1 == 0 {
                        bad += 1;
                        if bad < 10 {
                            eprintln!("frame {f} pattern {p:08b} sym {si} -> frame {g} {q:08b}");
                        }
                    }
                }
            }
        }
        assert_eq!(bad, 0);
    }
}
