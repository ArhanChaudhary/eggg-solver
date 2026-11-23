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
pub(crate) fn pext8_table() -> Box<[[u8; 256]; 256]> {
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
    /// NEON: per class, the 16 positions outside its representative's U/D mask, in order
    /// (see [`Staging::fast_index_flags`]).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    rem_pos: Vec<[u8; 16]>,
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
            stab,
            sym_perm,
            sym_gather,
            move_perm,
            unl,
            unl_inv,
            rank_t,
            #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
            pext8: pext8_table(),
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            rem_pos: class_rep
                .iter()
                .map(|&r| {
                    let mut p = (0..24u8).filter(|&i| r >> i & 1 == 0);
                    std::array::from_fn(|_| p.next().unwrap())
                })
                .collect(),
            class_rep,
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
        let g = &self.sym_gather[(info & 255) as usize];
        // NEON: the R/L flags at the image's positions outside the class representative's
        // U/D mask, in order, by two table lookups (instead of two movemasks and a pext).
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        let b = unsafe {
            use std::arch::aarch64::*;
            let ld = |v: &V32| uint8x16x2_t(vld1q_u8(v.0.as_ptr()), vld1q_u8(v.0.as_ptr().add(16)));
            let rp = vld1q_u8(self.rem_pos[class].as_ptr());
            let h = vqtbl2q_u8(ld(flags), vqtbl2q_u8(ld(g), rp));
            const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let m = vandq_u8(vcltzq_s8(vreinterpretq_s8_u8(h)), vld1q_u8(W.as_ptr()));
            let m = vpaddq_u8(m, m);
            let m = vpaddq_u8(m, m);
            let m = vpaddq_u8(m, m);
            vgetq_lane_u16::<0>(vreinterpretq_u16_u8(m))
        };
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        let b = {
            let h = flags.gather(g);
            let b = crate::simd::movemask(&h);
            let rem = crate::simd::movemask(&crate::simd::add(&h, &h));
            self.pext24(b, rem) as u16
        };
        (class * N_SPLIT + self.unl[b as usize] as usize) as u64
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

    /// [`Staging::big_index`] from the state's centre flags (see [`P1State::centre_flags`]).
    #[inline(always)]
    pub(crate) fn big_index_flags(&self, g: &V32) -> u64 {
        // NEON: both position masks by one pairwise-add chain, and the bit counts of their
        // bytes by one count.
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            use std::arch::aarch64::*;
            const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let w = vld1q_u8(W.as_ptr());
            let (a0, a1) = (vld1q_u8(g.0.as_ptr()), vld1q_u8(g.0.as_ptr().add(16)));
            let r0 = vandq_u8(vcltzq_s8(vreinterpretq_s8_u8(a0)), w);
            let r1 = vandq_u8(vcltzq_s8(vreinterpretq_s8_u8(a1)), w);
            let f = vdupq_n_u8(0x40);
            let n0 = vandq_u8(vtstq_u8(a0, f), w);
            let n1 = vandq_u8(vtstq_u8(a1, f), w);
            let z = vpaddq_u8(vpaddq_u8(r0, r1), vpaddq_u8(n0, n1));
            // Bytes 0..3: the R/L mask; bytes 4..7: the mask of the other colours.
            let z = vget_low_u8(vpaddq_u8(z, z));
            let x = vget_lane_u64::<0>(vreinterpret_u64_u8(z));
            let c = vget_lane_u64::<0>(vreinterpret_u64_u8(vcnt_u8(z)));
            let (rl, nud) = (x as u32, (x >> 32) as u32);
            let ud = !nud & MASK24;
            // Bit counts of the bytes of `nud` (each at most 8; U/D's are 8 minus them).
            let (n0, n1) = ((c >> 32 & 255) as u32, (c >> 40 & 255) as u32);
            debug_assert!(n0 <= 8 && n1 <= 8);
            let (c0, c1) = ((8 - n0) as usize, (16 - n0 - n1) as usize);
            let rank = self.rank_t[0][0][(ud & 255) as usize]
                + self.rank_t.get_unchecked(1).get_unchecked(c0)[(ud >> 8 & 255) as usize]
                + self.rank_t.get_unchecked(2).get_unchecked(c1)[(ud >> 16 & 255) as usize];
            let t = &self.pext8;
            let split = t[(nud & 255) as usize][(rl & 255) as usize] as u32
                | (t[(nud >> 8 & 255) as usize][(rl >> 8 & 255) as usize] as u32) << n0
                | (t[(nud >> 16 & 255) as usize][(rl >> 16 & 255) as usize] as u32) << (n0 + n1);
            rank as u64 * N_SPLIT as u64 + self.unl[split as u16 as usize] as u64
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            let (ud, nud, rl) = flag_masks(g);
            self.big_index_n(ud, nud, rl)
        }
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
    /// Per move, the frames (bit f) in which it is a phase-2 move.
    p2_frames: [u8; N_MOVES],
    /// Per set r of frames, the moves that are phase-2 moves in all frames but r's.
    all_p2: [u32; 8],
    /// Per move m, per position i: the source (as in `centre_idx[m]`) of the first position
    /// on i's axis (see [`P1Tables::staged_after`]).
    stage_rep: [V32; N_MOVES],
    /// The pattern table (searches per colour pair, see [`crate::p1pat`]).
    pub pat: Option<crate::p1pat::PatTable>,
    /// The move-mask table (with the large staging table, see [`crate::p1mask`]).
    pub mask: Option<crate::p1mask::MaskTable>,
    /// The wing lookahead of the batched nodes two moves from the end (see [`crate::wing2`]).
    pub wing2: Box<crate::wing2::Wing2>,
    /// The wing lookahead along the staging paths of tight nodes (see [`crate::wingpath`]),
    /// for the batched nodes at most its `kmax` moves from the end.
    pub wingk: Option<Box<crate::wingpath::WingPaths>>,
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
        let p2_frames = std::array::from_fn(|m| {
            (0..3)
                .filter(|&f| frames[f].p2_mask >> m & 1 == 1)
                .fold(0, |a, f| a | 1 << f)
        });
        let all_p2 = std::array::from_fn(|r| {
            (0..N_MOVES)
                .filter(|&m| (p2_frames[m] as usize | r) == 7)
                .fold(0, |a, m| a | 1 << m)
        });
        let stage_rep = std::array::from_fn(|m| {
            let mut v: V32 = centre_idx[m];
            for i in 0..24 {
                v.0[i] = centre_idx[m].0[4 * ((i / 4) % 3)];
            }
            v
        });
        let wing2 = Box::new(crate::wing2::Wing2::new(&frames, &wing_dst));
        P1Tables {
            staging,
            table,
            big: false,
            frames,
            centre_idx,
            wing_dst,
            p2_frames,
            all_p2,
            stage_rep,
            pat: None,
            mask: None,
            wing2,
            wingk: None,
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

    /// Are the centres staged after move `m` from `centres` (staging distance 0: on each
    /// axis, the 8 centres of one colour pair)?
    #[inline(always)]
    pub fn staged_after(&self, centres: &V32, m: usize) -> bool {
        debug_assert!(m < N_MOVES);
        const PAIR: V32 = V32([0xfe; 32]);
        let a = centres.gather(unsafe { self.centre_idx.get_unchecked(m) });
        let b = centres.gather(unsafe { self.stage_rep.get_unchecked(m) });
        crate::simd::is_zero(&crate::simd::and(&crate::simd::xor(&a, &b), &PAIR))
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
    /// Per node of the current round, what the pass ahead of the wing-path tests found (see
    /// [`Search1::run_batch_round`]).
    pre: Vec<Pre>,
    /// The wing-path lists of the children tested in the current round, located ahead (see
    /// [`crate::wingpath::WingPaths::locate`]).
    located: Vec<u64>,
}

/// What the pass ahead of a batch round's wing-path tests found about a node (see
/// [`Search1::run_batch_round`]).
#[derive(Clone, Copy)]
struct Pre {
    /// The children that can still reach an endpoint (u32::MAX: not computed, nor the rest).
    live: u32,
    /// Those whose wing-path lists are located in `Search1::located`, from `base`.
    tested: u32,
    base: u32,
    /// The node's decreasing and keeping moves (the latter only for a node with slack).
    dec: u32,
    eq: u32,
}

/// Prefetch the wing-path entries of the tight batch nodes' children a pass ahead
/// (`EGGG_P1_WKPF=0`: no).
static WK_PREFETCH: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("EGGG_P1_WKPF").map_or(true, |v| v != "0"));

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
    /// The exact distance: `left`, or less for a node with slack (reduced table, see
    /// `MaskTable::slack_batch`).
    d: u8,
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

/// Per wing parity (even, odd) and move, the last moves a leaf with that parity after the
/// move counts (see [`Search1::leaf`]).
const LEAF_MOVES: [[u8; N_MOVES]; 2] = {
    let mut t = [[0u8; N_MOVES]; 2];
    let mut m = 0;
    while m < N_MOVES {
        t[0][m] = (ALLOWED_AFTER[m] & !WIDE_QUARTER).count_ones() as u8;
        t[1][m] = (ALLOWED_AFTER[m] & WIDE_QUARTER).count_ones() as u8;
        m += 1;
    }
    t
};

/// The wide moves (Uw, Rw, Fw and their powers).
const WIDE_MOVES: u32 = 0x1ff << 18;

/// Per move (and none, [`N_MOVES`]), the moves on its axis (move m turns a face of axis
/// (m / 3) % 3): the moves that continue a trailing run of same-axis moves.
const SAME_AXIS: [u32; N_MOVES + 1] = {
    let mut t = [0u32; N_MOVES + 1];
    let mut a = 0;
    while a < N_MOVES {
        let mut m = 0;
        while m < N_MOVES {
            if (m / 3) % 3 == (a / 3) % 3 {
                t[a] |= 1 << m;
            }
            m += 1;
        }
        a += 1;
    }
    t
};

/// The trailing run of same-axis moves after move `m`, given the run `run` before it and the
/// move `last` before it.
#[inline(always)]
fn run_after(run: u32, last: usize, m: usize) -> u32 {
    if SAME_AXIS[last] >> m & 1 == 1 {
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
            pre: vec![],
            located: vec![],
        }
    }

    /// Queue a node with `left` moves left (>= 2) at distance `d` (= `left`, or less with the
    /// reduced table) for a batch: the state `s` after the search's path and `m` (if any),
    /// its mask table's first index `idx()` (computed only for a node that is queued).
    #[inline(always)]
    fn push_batch(
        &mut self,
        s: P1State,
        idx: impl FnOnce() -> u64,
        left: u8,
        last: usize,
        m: Option<u8>,
        d: u8,
    ) {
        if !self.wings_possible(&s, left, d) {
            return;
        }
        let idx = idx();
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
            d,
        });
    }

    /// Queue the child `c` (after move `m`, first index `idx()`, computed only if it is
    /// queued, distance `d`) of batch node `nd`, if it passes its wing tests (`loc`: its
    /// wing-path list, located ahead; see [`Search1::child_possible`]).
    #[inline(always)]
    fn push_child(
        &mut self,
        nd: &BNode,
        c: &P1State,
        idx: impl FnOnce() -> u64,
        m: usize,
        d: u8,
        loc: Option<u64>,
    ) {
        let ok = match loc {
            Some(l) => self
                .t
                .wingk
                .as_deref()
                .unwrap()
                .possible_at(l, &c.pos, c.parity),
            None => self.child_possible(c, nd.left - 1, d),
        };
        if !ok {
            return;
        }
        let idx = idx();
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
            d,
        });
    }

    /// Can a node queued for a batch (state `s`, `left` moves left, staging distance `d`)
    /// still reach an endpoint as far as its wings tell (see [`crate::wing2`])? Only nodes two
    /// moves from the end are tested, in the search over all pairs (where `d` is the staging
    /// distance); a node that fails has no endpoint below it and is not queued.
    #[inline(always)]
    fn wings_possible(&self, s: &P1State, left: u8, d: u8) -> bool {
        left != 2 || self.pair != ALL_PAIRS || self.t.wing2.possible(&s.pos, s.parity, d == 2)
    }

    /// Can a batch node's child (state `c`, `left` moves left, staging distance `d`) still
    /// reach an endpoint as far as its wings tell? A tight child (`d` = `left`) at most the
    /// wing-path table's `kmax` moves from the end, and a child with one move of slack at
    /// distance at most its `smax`, get its tests (see [`crate::wingpath`]), stronger than
    /// [`Search1::wings_possible`]; others that one.
    #[inline(always)]
    fn child_possible(&self, c: &P1State, left: u8, d: u8) -> bool {
        if self.pair == ALL_PAIRS
            && let Some(w) = self.t.wingk.as_deref()
        {
            if d == left && left <= w.kmax {
                let (ud, rl) = c.masks();
                return w.possible(ud, rl, &c.pos, c.parity);
            }
            if d == left && left == w.kmax + 1 && w.red.is_some() {
                let (ud, rl) = c.masks();
                return w.possible_red(ud, rl, &c.pos, c.parity);
            }
            if d + 1 == left && d <= w.smax {
                let (ud, rl) = c.masks();
                return w.possible_slack(ud, rl, &c.pos, c.parity);
            }
            if d + 2 == left && w.s2max.is_some_and(|x| d <= x) {
                let (ud, rl) = c.masks();
                return w.possible_slack2(ud, rl, &c.pos, c.parity);
            }
        }
        self.wings_possible(c, left, d)
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

    /// The rounds run in chunks of `MaskTable::chunk` nodes; once the next round holds that
    /// many nodes, their subtrees are searched before the rest of the round (depth first over
    /// chunks), and the paths of those subtrees' nodes are dropped from `batch_tree`: the
    /// buffers then stay at a few chunks per depth. (Whole rounds reached ~1M nodes, ~170 MB
    /// per search, at phase-1 length 17, ~2.7 GB at length 18, ~10x per length.)
    fn run_batch_rounds(&mut self) {
        let chunk = self.t.mask.as_ref().unwrap().chunk;
        while !self.batch.is_empty() {
            let mut cur = std::mem::take(&mut self.batch);
            std::mem::swap(&mut self.batch, &mut self.batch_next);
            self.batch.clear();
            // The ids of this round's children start here (`cur`'s and their ancestors' are
            // below it).
            let mark = self.batch_tree.len();
            let mut start = 0;
            while start < cur.len() {
                let end = (start + chunk).min(cur.len());
                self.run_batch_round(&mut cur[start..end]);
                if self.stop {
                    return;
                }
                start = end;
                if start < cur.len() && self.batch.len() >= chunk {
                    self.run_batch_rounds();
                    if self.stop {
                        return;
                    }
                    // Every node queued since `mark` has been searched.
                    self.batch_tree.truncate(mark);
                }
            }
            cur.clear();
            self.batch_next = cur;
        }
    }

    /// One round (or a chunk of one) of batch nodes `cur`: their children are queued in
    /// `self.batch`, or tested if they have one move left.
    fn run_batch_round(&mut self, cur: &mut [BNode]) {
        let t = self.t;
        let mt = t.mask.as_ref().unwrap();
        let reduced = mt.kind == crate::p1mask::MaskKind::Reduced;
        // The reduced table's viability bits are for the staging distance (all pairs).
        let viable = reduced && self.pair == ALL_PAIRS;
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
        // The nodes whose children get the wing-path tests: their live children now (u32::MAX
        // if not), and the tests split in three passes: the children's keys, their slots
        // prefetched (here); the slots read and the lists prefetched (next); the lists read
        // (when the children are queued).
        let mut pre = std::mem::take(&mut self.pre);
        let mut located = std::mem::take(&mut self.located);
        pre.clear();
        located.clear();
        if let Some(w) = t.wingk.as_deref().filter(|_| viable && *WK_PREFETCH) {
            use crate::wingpath::{ODD, SLACK, SLACK2};
            for nd in cur.iter() {
                let (d, left) = (nd.d, nd.left);
                let mut e = u32::MAX;
                let (mut dec_all, mut eq_all) = (0, 0);
                // The moves to children with the tests (tight ones, ones with slack 1).
                let mut tested = 0;
                // Those with slack 1, and with slack 2.
                let mut slack = 0;
                let mut slack2 = 0;
                if left < 3 {
                } else if d == left {
                    let red = w.red.is_some() && left - 1 == w.kmax + 1;
                    if (left - 1 <= w.kmax || red) && mt.viable(nd.idx, nd.parity) {
                        let dec = mt.get(t, nd.idx, nd.sym, &nd.state(), self.pair);
                        let a = dec & ALLOWED_AFTER[nd.last as usize];
                        e = self.live_children(nd, a, d, left, dec, 0);
                        tested = e;
                        dec_all = dec;
                    }
                } else if left - 1 <= w.kmax
                    || left - 2 <= w.smax
                    || w.s2max.is_some_and(|x| left - 3 <= x)
                {
                    let allowed = ALLOWED_AFTER[nd.last as usize];
                    (dec_all, eq_all) = mt.get_dec_eq(nd.idx, nd.sym);
                    let (dec, eq) = (dec_all & allowed, eq_all & allowed);
                    let a = if d + 1 < left { allowed } else { dec | eq };
                    e = self.live_children(nd, a, d, left, dec_all, eq_all);
                    // The children with no slack and with one move of slack (a decreasing
                    // move keeps the slack, a keeping move takes one, another two).
                    let inc = allowed & !(dec | eq);
                    let (tight, slack1) = match left - d {
                        1 => (eq, dec),
                        2 => (inc, eq),
                        3 => (0, inc),
                        _ => (0, 0),
                    };
                    slack = e & if left - 2 <= w.smax { slack1 } else { 0 };
                    if w.s2max.is_some_and(|x| left - 3 <= x) {
                        slack2 = e & match left - d {
                            2 => dec,
                            3 => eq,
                            4 => inc,
                            _ => 0,
                        };
                    }
                    tested = e & if left - 1 <= w.kmax { tight } else { 0 } | slack | slack2;
                }
                // The tests by the reduced lists (tight children one move beyond the table):
                // their class (and symmetry) is found here, once.
                let red = d == left && left - 1 == w.kmax + 1;
                let base = located.len() as u32;
                if tested != 0 {
                    let flags = nd.state().centre_flags();
                    let mut b = tested;
                    while b != 0 {
                        let m = b.trailing_zeros() as usize;
                        b &= b - 1;
                        let (ud, _, rl) = flag_masks(&unsafe { child_flags(&flags, m, t) });
                        let odd = nd.parity as u64 ^ (WIDE_QUARTER >> m & 1) as u64;
                        if red {
                            located.push(w.prefetch_red(ud, rl) | (odd * ODD));
                        } else {
                            let bits = (odd * ODD) | ((slack >> m & 1) as u64 * SLACK);
                            if slack2 >> m & 1 == 1 {
                                located.push(w.prefetch_key2(ud, rl) | bits | SLACK2);
                            } else {
                                located.push(w.prefetch_key(ud, rl) | bits);
                            }
                        }
                    }
                }
                pre.push(Pre {
                    live: e,
                    tested,
                    base,
                    dec: dec_all,
                    eq: eq_all,
                });
            }
            for x in located.iter_mut() {
                *x = w.locate(*x);
            }
        }
        self.pre = pre;
        self.located = located;
        for (ix, nd) in cur.iter().enumerate() {
            let s = nd.state();
            if nd.d < nd.left {
                // A node with slack (reduced table): its children's distances from its
                // own entry, as in `dfs_eq`; they are all queued, or with one move left
                // tested.
                let (d, left) = (nd.d, nd.left);
                let allowed = ALLOWED_AFTER[nd.last as usize];
                // The masks read by the pass ahead, if it did.
                let pre = self.pre.get(ix).copied().filter(|p| p.live != u32::MAX);
                let (dec_all, eq_all) = match pre {
                    Some(p) => (p.dec, p.eq),
                    None => mt.get_dec_eq(nd.idx, nd.sym),
                };
                let (dec, eq) = (dec_all & allowed, eq_all & allowed);
                self.nodes += allowed.count_ones() as u64;
                self.reads += 1;
                let mut a = if d + 1 < left { allowed } else { dec | eq };
                if viable {
                    // The children that cannot reach an endpoint (child bits) are not
                    // queued, nor tested (their last moves are counted).
                    let live = match pre {
                        Some(p) => p.live,
                        _ => self.live_children(nd, a, d, left, dec_all, eq_all),
                    };
                    if left == 2 {
                        let mut b = a & !live;
                        while b != 0 {
                            let m = b.trailing_zeros() as usize;
                            b &= b - 1;
                            let odd = nd.parity ^ (WIDE_QUARTER >> m & 1 == 1);
                            self.nodes += LEAF_MOVES[odd as usize][m] as u64;
                        }
                    }
                    a = live;
                }
                let flags = s.centre_flags();
                let (tested, base) = pre.map_or((0, 0), |p| (p.tested, p.base));
                while a != 0 {
                    let m = a.trailing_zeros() as usize;
                    a &= a - 1;
                    let dc = d + 1 - 2 * (dec >> m & 1) as u8 - (eq >> m & 1) as u8;
                    let c = s.apply(m, t);
                    if left == 2 {
                        let run = run_after(nd.run, nd.last as usize, m);
                        self.leaf(&c, dc, m, None, run, Some((nd.id, Some(m as u8))));
                        if self.stop {
                            self.batch.clear();
                            return;
                        }
                    } else {
                        let i = || {
                            let g = unsafe { child_flags(&flags, m, t) };
                            let nud = crate::simd::movemask(&crate::simd::add(&g, &g));
                            t.staging.rank24(!nud & MASK24) as u64
                        };
                        let loc = self.located(tested, base, m);
                        self.push_child(nd, &c, i, m, dc, loc);
                    }
                }
                continue;
            }
            // A node whose wing parity cannot reach an endpoint in moves left = distance
            // moves (its entry's viability bit, see `MaskTable::add_viable`) is dropped.
            // (The pass ahead read the entry of a node it found viable.)
            let pre = self.pre.get(ix).copied().filter(|p| p.live != u32::MAX);
            if viable && pre.is_none() && !mt.viable(nd.idx, nd.parity) {
                self.reads += 1;
                continue;
            }
            let dec = match pre {
                Some(p) => p.dec,
                None => mt.get(t, nd.idx, nd.sym, &s, self.pair),
            };
            let (live, tested, base) = match pre {
                Some(p) => (Some(p.live), p.tested, p.base),
                None => (None, 0, 0),
            };
            let mut a = dec & ALLOWED_AFTER[nd.last as usize];
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
                // Over all pairs, a child (at distance 1) whose wings have even parity has
                // no endpoint below it: its last move must not be a wide quarter turn, and
                // then cannot stage the centres or is a phase-2 move in every frame (see
                // `leaf`). Its last moves are counted, not tested.
                if self.pair == ALL_PAIRS {
                    let even = if s.parity {
                        WIDE_QUARTER
                    } else {
                        !WIDE_QUARTER
                    };
                    let mut b = a & even;
                    a &= !even;
                    while b != 0 {
                        let m = b.trailing_zeros() as usize;
                        b &= b - 1;
                        self.nodes += LEAF_MOVES[0][m] as u64;
                    }
                }
                // Nor has one that the child bits rule out (odd parity: counted too).
                if viable {
                    let live = self.live_children(nd, a, nd.d, nd.left, dec, 0);
                    let mut b = a & !live;
                    a = live;
                    while b != 0 {
                        let m = b.trailing_zeros() as usize;
                        b &= b - 1;
                        self.nodes += LEAF_MOVES[1][m] as u64;
                    }
                }
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
                    let i = || {
                        let g = unsafe { child_flags(&flags, m, t) };
                        t.staging.rank24(crate::simd::movemask(&g)) as u64
                    };
                    let c = s.apply(m, t);
                    self.push_child(nd, &c, i, m, nd.left - 1, None);
                }
            } else {
                if viable {
                    // The children that cannot reach an endpoint are not queued (the
                    // reduced table's child bits; they are still counted as nodes).
                    a = match live {
                        Some(a) => a,
                        None => self.live_children(nd, a, nd.d, nd.left, dec, 0),
                    };
                }
                let flags = s.centre_flags();
                while a != 0 {
                    let m = a.trailing_zeros() as usize;
                    a &= a - 1;
                    let i = || {
                        let g = unsafe { child_flags(&flags, m, t) };
                        // The reduced table needs only the rank of the U/D positions.
                        if reduced {
                            let nud = crate::simd::movemask(&crate::simd::add(&g, &g));
                            t.staging.rank24(!nud & MASK24) as u64
                        } else {
                            t.staging.big_index_flags(&g)
                        }
                    };
                    let c = s.apply(m, t);
                    let loc = self.located(tested, base, m);
                    self.push_child(nd, &c, i, m, nd.left - 1, loc);
                }
            }
        }
    }

    /// The located wing-path list of the child via move `m` of a node of the current round
    /// whose children `tested` have theirs in `self.located` from `base` (see
    /// [`Search1::run_batch_round`]).
    #[inline(always)]
    fn located(&self, tested: u32, base: u32, m: usize) -> Option<u64> {
        (tested >> m & 1 == 1).then(|| {
            let i = base as usize + (tested & ((1 << m) - 1)).count_ones() as usize;
            debug_assert!(i < self.located.len());
            // SAFETY: one entry per tested child.
            unsafe { *self.located.get_unchecked(i) }
        })
    }

    /// The children among `a` (moves of batch node `nd`, at distance `d` with `left` moves
    /// left, its decreasing and keeping moves `dec` and `eq`, all of each) without those at
    /// distance = moves left after the move that cannot reach an endpoint (the reduced
    /// table's child bits, see [`crate::p1mask::MaskTable::child_live`]): the children via a
    /// decreasing move if `d` = `left`, via a keeping move if `d` + 1 = `left`, via any other
    /// move if `d` + 2 = `left`. A child's decreasing move on the move's axis is then
    /// respectively a decreasing move, a decreasing or keeping move, or any move of the node.
    #[inline(always)]
    fn live_children(&self, nd: &BNode, a: u32, d: u8, left: u8, dec: u32, eq: u32) -> u32 {
        use crate::p1mask::MaskTable;
        let (tight, pre) = if d == left {
            (dec, dec)
        } else if d + 1 == left {
            (eq, dec | eq)
        } else if d + 2 == left {
            (!(dec | eq), u32::MAX)
        } else {
            return a;
        };
        let mt = self.t.mask.as_ref().unwrap();
        let bits = mt.child_bits(nd.idx, nd.parity);
        let stored = mt.stored_moves(nd.sym);
        let mut x = a & tight;
        let mut out = a;
        while x != 0 {
            let m = x.trailing_zeros() as usize;
            x &= x - 1;
            if !MaskTable::child_live(bits, stored, m, pre) {
                out &= !(1 << m);
            }
        }
        out
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
        } else if let Some(mt) = self
            .t
            .mask
            .as_ref()
            .filter(|m| m.kind == crate::p1mask::MaskKind::Reduced)
        {
            let r = mt.resolve(self.t, mt.pre_index(self.t, s, ALL_PAIRS), s, ALL_PAIRS);
            self.dfs_eq(s, d, left, last, r);
            if !self.stop {
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

    /// [`Search1::dfs`] over all pairs with the reduced move-mask table. A node with slack
    /// (distance `d` < moves left `left`) has its children's exact distances from its own
    /// entry (`r`: its reduced index and symmetry, the entry prefetched): the moves that
    /// decrease and that keep the distance, and the others increase it. So it reads one entry
    /// where [`Search1::dfs_t`] reads one per child. The children are searched as there, in
    /// move order: those at distance = moves left (>= 2) are queued for a batch, the others
    /// searched depth-first, their entries resolved and prefetched first.
    fn dfs_eq(&mut self, s: &P1State, d: u8, left: u8, last: usize, r: (u64, u8)) {
        if left == 0 {
            if d == 0 {
                let frames = self.t.endpoint_frames(s, trailing_run(&self.path));
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
        let t = self.t;
        let st = &t.staging;
        let mt = t.mask.as_ref().unwrap();
        if d >= left {
            if d == left {
                self.push_batch(*s, || st.rank24(s.masks().0) as u64, left, last, None, d);
            }
            return;
        }
        let allowed = ALLOWED_AFTER[last];
        let (dec, eq) = mt.get_dec_eq(r.0, r.1);
        let (dec, eq) = (dec & allowed, eq & allowed);
        self.nodes += allowed.count_ones() as u64;
        self.reads += 1;
        if self
            .abort
            .is_some_and(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        {
            self.stop = true;
            return;
        }
        // The children that can still reach the goal: at distance d - 1 or d, and at d + 1
        // if that is less than the moves left after the move.
        let visit = if d + 1 < left { allowed } else { dec | eq };
        // Those at distance = moves left after the move (batched if >= the batch minimum).
        let tight = if d + 1 == left {
            eq
        } else if d + 2 == left {
            visit & !(dec | eq)
        } else {
            0
        };
        let mut batched = if left > self.batch_min_left() {
            tight
        } else {
            0
        };
        // With few moves left, the children with slack are batched too: their entries are
        // then read a round after their prefetch (and the batches see their subtrees' nodes
        // at distance = moves left sooner).
        if left >= 3 && left - 1 <= mt.slack_batch {
            batched = visit;
        }
        // The children searched depth-first with at least 2 moves left: their entries'
        // indices (class info, prefetched, then the entry, prefetched), by move.
        let mut rs = [(0u64, 0u8); N_MOVES];
        if left >= 3 {
            let flags = s.centre_flags();
            let mut infos: [*const u32; N_MOVES] = [std::ptr::null(); N_MOVES];
            let mut gs = [V32::ZERO; N_MOVES];
            let mut a = visit & !batched;
            while a != 0 {
                let m = a.trailing_zeros() as usize;
                a &= a - 1;
                // SAFETY: m < N_MOVES.
                let g = unsafe { child_flags(&flags, m, t) };
                let nud = crate::simd::movemask(&crate::simd::add(&g, &g));
                let p = st.info(!nud & MASK24);
                crate::table::prefetch(p);
                unsafe {
                    *infos.get_unchecked_mut(m) = p;
                    *gs.get_unchecked_mut(m) = g;
                }
            }
            let mut a = visit & !batched;
            while a != 0 {
                let m = a.trailing_zeros() as usize;
                a &= a - 1;
                unsafe {
                    let info = **infos.get_unchecked(m);
                    let j = st.fast_index_flags(info, gs.get_unchecked(m));
                    mt.prefetch_entry(j);
                    *rs.get_unchecked_mut(m) = (j, info as u8);
                }
            }
        }
        let mut a = visit;
        while a != 0 {
            let m = a.trailing_zeros() as usize;
            a &= a - 1;
            // d - 1, d or d + 1 (the masks are disjoint).
            let dc = d + 1 - 2 * (dec >> m & 1) as u8 - (eq >> m & 1) as u8;
            let c = s.apply(m, t);
            if batched >> m & 1 == 1 {
                let i = || st.rank24(c.masks().0) as u64;
                self.push_batch(c, i, left - 1, m, Some(m as u8), dc);
                if self.batch.len() >= mt.batch {
                    self.run_batch();
                }
            } else {
                self.path.push(m as u8);
                if left == 2 {
                    self.leaf(&c, dc, m, None, trailing_run(&self.path), None);
                } else {
                    self.dfs_eq(&c, dc, left - 1, m, rs[m]);
                }
                self.path.pop();
            }
            if self.stop {
                return;
            }
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
        let allowed = ALLOWED_AFTER[last]
            & staging.unwrap_or(u32::MAX)
            & if s.parity {
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
        // Every allowed move is a node (the ones not tested after a stop are taken back).
        self.nodes += allowed.count_ones() as u64;
        // The moves that continue the run (on the last move's axis).
        let same = SAME_AXIS[last];
        let t = self.t;
        let pos = s.pos;
        // A move that is a phase-2 move in every frame (with the run it continues) ends no
        // endpoint: it is counted, not tested (the half turns, for one).
        let mut test = allowed & !(t.all_p2[0] | same & t.all_p2[run_frames as usize]);
        // Without the staging moves from a mask, the search over all pairs tests that the move
        // stages the centres first (the cheaper and less selective test; no table read). From
        // unstaged centres (d >= 1) only a wide move can stage them: an outer move keeps every
        // face's centres on that face.
        let check_staged = staging.is_none() && self.pair == ALL_PAIRS;
        if check_staged && d >= 1 {
            test &= WIDE_MOVES;
        }
        let centres = s.centres;
        while test != 0 {
            let m = test.trailing_zeros() as usize;
            test &= test - 1;
            if check_staged && !t.staged_after(&centres, m) {
                continue;
            }
            let excl = unsafe { *t.p2_frames.get_unchecked(m) } as u32
                | run_frames as u32 & 0u32.wrapping_sub(same >> m & 1);
            let p = unsafe { t.wing_dst.get_unchecked(m) }.gather(&pos);
            let mut frames = 0u8;
            for (f, fr) in t.frames.iter().enumerate() {
                if excl >> f & 1 == 0 && fr.wings_ok(&p) {
                    frames |= 1 << f;
                }
            }
            if frames == 0 {
                continue;
            }
            let c = s.apply(m, self.t);
            if staging.is_none() && !check_staged {
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
                    // The moves after m were not tested.
                    self.nodes -= (allowed >> m >> 1).count_ones() as u64;
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
            self.push_batch(*s, || i, left, last, None, left);
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
                let i = t
                    .staging
                    .big_index_flags(&unsafe { child_flags(&flags, m, t) });
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
                    let pair = self.pair;
                    let i = || {
                        if K == K_MASK {
                            t.mask.as_ref().unwrap().pre_index_of_big(idx[k])
                        } else {
                            t.mask.as_ref().unwrap().pre_index(t, &c, pair)
                        }
                    };
                    self.push_batch(c, i, dc, m, Some(m as u8), dc);
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
                    assert_eq!(t.staging.big_index_flags(&g), t.staging.big_index(ud, rl));
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

    /// The batched search with move masks (computed, and the reduced table with its
    /// viability bits and equal masks, slack nodes batched or not, batch rounds in chunks of
    /// any size), and the per-pair searches
    /// with the pattern table, find exactly the
    /// plain search's (endpoint, frame) pairs (needs the large staging table, and for the
    /// pairs the reduced pattern table, in ./tables; skipped without them; the reduced mask
    /// table is loaded, or made and saved).
    #[test]
    fn masked_and_pair_searches_find_the_same_endpoints() {
        let dir = std::path::Path::new("tables");
        if !dir.join("p1-staging-big.tab").exists() {
            return;
        }
        let mut t = crate::solver::load_p1(dir, true, 8, false);
        let mut reduced = Some(crate::p1mask::load(dir, &t, false, false));
        let with_pat = dir.join("p1-pat.tab").exists();
        if with_pat {
            crate::solver::load_pat(&mut t, dir, false, 8, false);
        }
        let mut rng = rand::thread_rng();
        for _ in 0..3 {
            let root = P1State::from_cube(&Cube::random(&mut rng));
            let d0 = t.distance(&root);
            for len in d0..d0 + 3 {
                t.mask = None;
                let want = endpoint_set(&t, &root, len, false);
                t.mask = Some(crate::p1mask::MaskTable::virtual_table());
                assert_eq!(endpoint_set(&t, &root, len, false), want);
                // Slack nodes batched with at most k moves left (none, the default, more).
                for k in [0, crate::p1mask::SLACK_BATCH, 10] {
                    t.mask = reduced.take();
                    t.mask.as_mut().unwrap().slack_batch = k;
                    assert_eq!(endpoint_set(&t, &root, len, false), want);
                    reduced = t.mask.take();
                }
                // Batch rounds in chunks of one node and of three (depth first throughout).
                for chunk in [1, 3] {
                    t.mask = reduced.take();
                    let mt = t.mask.as_mut().unwrap();
                    (mt.slack_batch, mt.chunk) = (crate::p1mask::SLACK_BATCH, chunk);
                    assert_eq!(endpoint_set(&t, &root, len, false), want);
                    t.mask.as_mut().unwrap().chunk = crate::p1mask::CHUNK;
                    reduced = t.mask.take();
                }
                if with_pat {
                    assert_eq!(endpoint_set(&t, &root, len, true), want);
                }
            }
        }
    }

    /// The masked search over all pairs finds the same endpoints, in the same order, with and
    /// without the wing-path lookahead (largest distance 3 to 5, the last with the reduced
    /// lists at 6), on random roots and on roots a few moves from endpoints, at their
    /// distance and up to 3 moves more (needs the reduced mask table in ./tables, loaded or
    /// made; the small staging table likewise).
    #[test]
    fn wing_paths_keep_the_endpoints() {
        let dir = std::path::Path::new("tables");
        let mut t = crate::solver::load_p1(dir, false, 8, false);
        t.mask = Some(crate::p1mask::load(dir, &t, false, false));
        let mut ws: Vec<_> = (3..=5)
            .map(|k| {
                Some(Box::new(crate::wingpath::WingPaths::new(
                    &t.frames,
                    k,
                    k - 1,
                    k == 5,
                )))
            })
            .collect();
        let run = |t: &P1Tables, root: &P1State, len: u8| {
            let mut found = vec![];
            let mut s = Search1::new(t, |p: &[u8], f| {
                found.push((p.to_vec(), f));
                true
            });
            s.dfs(root, t.distance(root), len, N_MOVES);
            let nodes = s.nodes;
            drop(s);
            (found, nodes)
        };
        let mut rng = <rand_xoshiro::Xoshiro256PlusPlus as rand::SeedableRng>::seed_from_u64(5);
        let mut found = 0;
        for i in 0..8 {
            let root = if i < 4 {
                P1State::from_cube(&Cube::random(&mut rng))
            } else {
                // An endpoint (solved cube) undone by 8 random moves.
                let mut s = P1State::from_cube(&Cube::SOLVED);
                for _ in 0..8 {
                    s = s.apply(rand::Rng::gen_range(&mut rng, 0..N_MOVES), &t);
                }
                s
            };
            let d0 = t.distance(&root);
            for len in d0..=d0 + 3 {
                t.wingk = None;
                let (want, nodes) = run(&t, &root, len);
                found += want.len();
                for w in ws.iter_mut() {
                    std::mem::swap(&mut t.wingk, w);
                    let (got, n) = run(&t, &root, len);
                    std::mem::swap(&mut t.wingk, w);
                    assert_eq!(got, want, "root {i} len {len}");
                    assert!(n <= nodes);
                }
            }
        }
        assert!(found > 0);
    }

    /// [`P1Tables::staged_after`] agrees with the definition (and with the staging table's
    /// goal, distance 0, where the large table is in ./tables).
    #[test]
    fn staged_after_matches_definition() {
        let t = P1Tables::with_table(Staging::new(), Mod3Table::from_depths(&[0]));
        let mut rng = rand::thread_rng();
        let mut seen = [0u32; 2];
        for k in 0..20000 {
            let mut s = P1State::from_cube(&Cube::SOLVED);
            for j in 0..k % 4 {
                s = s.apply((k * 11 + j * 7) % N_MOVES, &t);
            }
            if k % 3 == 0 {
                s = P1State::from_cube(&Cube::random(&mut rng));
            }
            for m in 0..N_MOVES {
                let c = s.apply(m, &t);
                let want = (0..3).all(|a| {
                    let p: Vec<u8> = (0..24)
                        .filter(|&i| (i / 4) % 3 == a)
                        .map(|i| c.centres.0[i] >> 1)
                        .collect();
                    p.iter().all(|&x| x == p[0])
                });
                assert_eq!(t.staged_after(&s.centres, m), want);
                seen[want as usize] += 1;
            }
        }
        assert!(seen[0] > 0 && seen[1] > 0);
        // The leaf's shortcuts: the wide half turns are phase-2 moves in every frame, and an
        // outer move never changes whether the centres are staged.
        assert_eq!(WIDE_MOVES & !WIDE_QUARTER & !t.all_p2[0], 0);
        let s = P1State::from_cube(&Cube::random(&mut rng));
        for m in (0..N_MOVES).filter(|&m| WIDE_MOVES >> m & 1 == 0) {
            let c = s.apply(m, &t);
            for a in 0..6 {
                let face = |x: &P1State| {
                    let mut v: Vec<u8> = x.centres.0[4 * a..4 * a + 4].to_vec();
                    v.sort();
                    v
                };
                assert_eq!(face(&s), face(&c));
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
