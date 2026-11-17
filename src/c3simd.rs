//! A 3x3x3 in one 32-byte vector, with moves and H48 coordinates computed by byte shuffles.
//!
//! Layout (vcube/egg-solver style): bytes 0..8 hold the corner at each position as
//! `piece | twist << 4`, bytes 16..28 the edge at each position as `piece | flip << 4`. Every
//! operation is per 16-byte lane, like `pshufb`.

use crate::cube3::{Cube3, MOVES3, N_MOVES3};
use crate::h48::{Coords, SLICE, TETRAD};
use std::sync::LazyLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C, align(32))]
pub struct C3(pub [u8; 32]);

/// Byte shuffle with `pshufb` semantics in each 16-byte lane.
#[inline(always)]
pub fn shuf(a: &C3, idx: &C3) -> C3 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let r = _mm256_shuffle_epi8(
            _mm256_load_si256(a.0.as_ptr() as *const __m256i),
            _mm256_load_si256(idx.0.as_ptr() as *const __m256i),
        );
        let mut o = C3([0; 32]);
        _mm256_store_si256(o.0.as_mut_ptr() as *mut __m256i, r);
        o
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut o = [0u8; 32];
        for i in 0..32 {
            let lane = i & 16;
            let x = idx.0[i];
            o[i] = if x & 0x80 != 0 {
                0
            } else {
                a.0[lane + (x & 15) as usize]
            };
        }
        C3(o)
    }
}

#[inline(always)]
fn zip(a: &C3, b: &C3, f: impl Fn(u8, u8) -> u8) -> C3 {
    let mut o = [0u8; 32];
    for i in 0..32 {
        o[i] = f(a.0[i], b.0[i]);
    }
    C3(o)
}

/// Bit 7 of every byte, per lane (bits 0..16 = low lane, 16..32 = high lane).
#[inline(always)]
pub fn movemask(a: &C3) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        _mm256_movemask_epi8(_mm256_load_si256(a.0.as_ptr() as *const __m256i)) as u32
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut m = 0;
        for i in 0..32 {
            m |= ((a.0[i] >> 7) as u32) << i;
        }
        m
    }
}

/// Per-move constants.
pub struct MoveC {
    /// Position gather indices (corners in lane 0, edges in lane 1).
    pub idx: C3,
    /// Twist << 4 to add (corners), flip << 4 to xor (edges).
    pub add: C3,
    pub xor: C3,
    /// Inverse-side relabelling: indexed by position, gives the new position with the negated
    /// twist (corners) or flip (edges) of the destination.
    pub rel: C3,
}

pub static MOVEC: LazyLock<Vec<MoveC>> = LazyLock::new(|| {
    (0..N_MOVES3)
        .map(|m| {
            let mv = &MOVES3[m];
            let mut c = MoveC {
                idx: C3([0x80; 32]),
                add: C3([0; 32]),
                xor: C3([0; 32]),
                rel: C3([0x80; 32]),
            };
            for t in 0..8 {
                c.idx.0[t] = mv.corner_src[t];
                c.add.0[t] = mv.corner_twist[t] << 4;
                let s = mv.corner_src[t] as usize;
                // A piece at position s goes to t with twist tw[t] added.
                c.rel.0[s] = t as u8 | ((3 - mv.corner_twist[t]) % 3) << 4;
            }
            for t in 0..12 {
                c.idx.0[16 + t] = mv.edge_src[t];
                c.xor.0[16 + t] = mv.edge_flip[t] << 4;
                let s = mv.edge_src[t] as usize;
                c.rel.0[16 + s] = t as u8 | mv.edge_flip[t] << 4;
            }
            c
        })
        .collect()
});

const SUB48: C3 = {
    let mut a = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        a[i] = 48;
        i += 1;
    }
    C3(a)
};

impl C3 {
    pub fn from_cube(x: &Cube3) -> C3 {
        let mut v = [0u8; 32];
        for i in 0..8 {
            v[i] = x.cp[i] | x.co[i] << 4;
        }
        for i in 0..12 {
            v[16 + i] = x.ep[i] | x.eo[i] << 4;
        }
        C3(v)
    }

    pub fn to_cube(&self) -> Cube3 {
        let mut x = Cube3::SOLVED;
        for i in 0..8 {
            x.cp[i] = self.0[i] & 15;
            x.co[i] = self.0[i] >> 4;
        }
        for i in 0..12 {
            x.ep[i] = self.0[16 + i] & 15;
            x.eo[i] = self.0[16 + i] >> 4;
        }
        x
    }

    pub const SOLVED: C3 = {
        let mut a = [0u8; 32];
        let mut i = 0;
        while i < 8 {
            a[i] = i as u8;
            i += 1;
        }
        let mut j = 0;
        while j < 12 {
            a[16 + j] = j as u8;
            j += 1;
        }
        C3(a)
    };

    /// Mod-3 reduction of corner twists in 0..5 (as `twist << 4`).
    #[inline(always)]
    fn fix_twist(v: C3) -> C3 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use std::arch::x86_64::*;
            let a = _mm256_load_si256(v.0.as_ptr() as *const __m256i);
            let b = _mm256_sub_epi8(a, _mm256_load_si256(SUB48.0.as_ptr() as *const __m256i));
            let r = _mm256_min_epu8(a, b);
            let mut o = C3([0; 32]);
            _mm256_store_si256(o.0.as_mut_ptr() as *mut __m256i, r);
            o
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            zip(&v, &SUB48, |a, s| a.min(a.wrapping_sub(s)))
        }
    }

    /// The cube after move m (applied on the right: positions move).
    #[inline(always)]
    pub fn apply(&self, m: usize) -> C3 {
        let c = &MOVEC[m];
        let g = shuf(self, &c.idx);
        let a = Self::fix_twist(add(&g, &c.add));
        xor(&a, &c.xor)
    }

    /// For the inverse cube y = x^-1: the inverse of x after move m, i.e. m^-1 * y.
    #[inline(always)]
    pub fn apply_inv(&self, m: usize) -> C3 {
        let c = &MOVEC[m];
        let r = shuf(&c.rel, self);
        // Corners: add the old twist; edges: xor the old flip.
        let tw = and(self, &TWMASK);
        let fl = and(self, &FLMASK);
        let a = Self::fix_twist(add(&r, &tw));
        and(&xor(&a, &fl), &VALID)
    }

    #[inline(always)]
    pub fn is_solved(&self) -> bool {
        *self == C3::SOLVED
    }

    pub fn inverse(&self) -> C3 {
        C3::from_cube(&self.to_cube().inverse())
    }
}

const VALID: C3 = {
    let mut a = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        if i < 8 || (i >= 16 && i < 28) {
            a[i] = 0xff;
        }
        i += 1;
    }
    C3(a)
};
const TWMASK: C3 = {
    let mut a = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        a[i] = 0x30;
        i += 1;
    }
    C3(a)
};
const FLMASK: C3 = {
    let mut a = [0u8; 32];
    let mut i = 16;
    while i < 28 {
        a[i] = 0x10;
        i += 1;
    }
    C3(a)
};

#[inline(always)]
fn add(a: &C3, b: &C3) -> C3 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let r = _mm256_add_epi8(
            _mm256_load_si256(a.0.as_ptr() as *const __m256i),
            _mm256_load_si256(b.0.as_ptr() as *const __m256i),
        );
        let mut o = C3([0; 32]);
        _mm256_store_si256(o.0.as_mut_ptr() as *mut __m256i, r);
        o
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        zip(a, b, |x, y| x.wrapping_add(y))
    }
}

#[inline(always)]
fn xor(a: &C3, b: &C3) -> C3 {
    zip(a, b, |x, y| x ^ y)
}

#[inline(always)]
fn and(a: &C3, b: &C3) -> C3 {
    zip(a, b, |x, y| x & y)
}

/// Shuffle-based H48 coordinate tables.
pub struct H48Fast {
    /// TETRAD flag (0x80) of a corner piece, indexed by piece; lane 0 only.
    tetrad: C3,
    /// co index from the two twist bit masks: co_t[lo7 | hi7 << 7].
    co_t: Vec<u16>,
    /// Per symmetry: piece -> E flag (0x80) | M flag (0x40) | flip contribution (0x10), lane 1,
    /// the position flip contribution (lane 1), and the position permutation gather.
    pt: Vec<C3>,
    pb: Vec<C3>,
    pperm: Vec<C3>,
    rank495: Vec<u16>,
    rank70: [u8; 256],
}

impl H48Fast {
    pub fn new(co: &Coords) -> H48Fast {
        let mut tetrad = C3([0; 32]);
        for p in 0..8 {
            tetrad.0[p] = TETRAD[p] << 7;
        }
        let mut co_t = vec![0u16; 1 << 14];
        for lo in 0..128usize {
            for hi in 0..128usize {
                if lo & hi != 0 {
                    continue;
                }
                let mut idx = 0usize;
                for i in (0..7).rev() {
                    let t = (lo >> i & 1) + 2 * (hi >> i & 1);
                    idx = idx * 3 + t;
                }
                co_t[lo | hi << 7] = idx as u16;
            }
        }
        let s = &co.syms;
        let mut pt = vec![];
        let mut pb = vec![];
        let mut pperm = vec![];
        for k in 0..48 {
            let mut a = C3([0; 32]);
            let mut b = C3([0; 32]);
            let mut g = C3([0x80; 32]);
            for p in 0..12 {
                let sp = s.eslot[k][p] as usize;
                let sl = SLICE[sp];
                a.0[16 + p] = ((sl == 2) as u8) << 7 | ((sl == 0) as u8) << 6 | s.ea[k][p] << 4;
                b.0[16 + p] = s.eb[k][p] << 4;
                // position p moves to s(p): gather index of the new position s(p) is p.
                g.0[16 + sp] = p as u8;
            }
            pt.push(a);
            pb.push(b);
            pperm.push(g);
        }
        let mut rank495 = vec![0u16; 4096];
        for m in 0u32..4096 {
            if m.count_ones() == 4 {
                rank495[m as usize] = crate::comb::comb_rank(m) as u16;
            }
        }
        let mut rank70 = [0u8; 256];
        for m in 0u32..256 {
            if m.count_ones() == 4 {
                rank70[m as usize] = crate::comb::comb_rank(m) as u8;
            }
        }
        H48Fast {
            tetrad,
            co_t,
            pt,
            pb,
            pperm,
            rank495,
            rank70,
        }
    }

    /// cocsep index of a cube.
    #[inline(always)]
    pub fn cocsep(&self, v: &C3) -> usize {
        let lo = movemask(&shl(v, 3)) & 0x7f;
        let hi = movemask(&shl(v, 2)) & 0x7f;
        let co = self.co_t[(lo | hi << 7) as usize] as usize;
        let sep = movemask(&shuf(&self.tetrad, v)) & 0xff;
        co * 70 + self.rank70[sep as usize] as usize
    }

    /// (eslot, msplit, eo11) of the cube conjugated by symmetry k.
    #[inline(always)]
    pub fn edges(&self, v: &C3, k: usize) -> (u32, u32, u32) {
        let z = shuf(&self.pt[k], v);
        let z = xor(&xor(&z, &and(v, &FLMASK)), &self.pb[k]);
        let z = shuf(&z, &self.pperm[k]);
        let e = movemask(&z) >> 16 & 0xfff;
        let m = movemask(&shl(&z, 1)) >> 16 & 0xfff;
        let o = movemask(&shl(&z, 3)) >> 16 & 0x7ff;
        let eslot = self.rank495[e as usize] as u32;
        let msplit = self.rank70[crate::phase1::pext(m, !e & 0xfff) as usize] as u32;
        (eslot, msplit, o)
    }
}

/// Shift every byte left by `n` bits (as 16-bit lanes; the bits crossing bytes do not matter
/// for movemask of bit 7 when the source byte's high bits are what we want).
#[inline(always)]
fn shl(a: &C3, n: u32) -> C3 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use std::arch::x86_64::*;
        let v = _mm256_load_si256(a.0.as_ptr() as *const __m256i);
        let r = match n {
            1 => _mm256_slli_epi16(v, 1),
            2 => _mm256_slli_epi16(v, 2),
            3 => _mm256_slli_epi16(v, 3),
            _ => v,
        };
        let mut o = C3([0; 32]);
        _mm256_store_si256(o.0.as_mut_ptr() as *mut __m256i, r);
        o
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut o = [0u8; 32];
        for i in 0..32 {
            o[i] = a.0[i] << n;
        }
        C3(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn simd_moves_match() {
        let mut rng = rand::thread_rng();
        let mut x = Cube3::SOLVED;
        let mut v = C3::SOLVED;
        let mut y = C3::SOLVED;
        for _ in 0..200 {
            let m = rng.gen_range(0..18);
            x = x.apply(m);
            v = v.apply(m);
            y = y.apply_inv(m);
            assert_eq!(v.to_cube(), x);
            assert_eq!(y.to_cube(), x.inverse());
        }
    }

    #[test]
    fn simd_coords_match() {
        let co = Coords::new();
        let f = H48Fast::new(&co);
        let mut rng = rand::thread_rng();
        let mut x = Cube3::SOLVED;
        for _ in 0..200 {
            x = x.apply(rng.gen_range(0..18));
            let v = C3::from_cube(&x);
            assert_eq!(f.cocsep(&v), co.cocsep_of(&x));
            for k in 0..48 {
                assert_eq!(f.edges(&v, k), co.edge_part_conj(&x, k));
            }
        }
    }
}
