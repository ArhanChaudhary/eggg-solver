//! 32-byte vectors and the few byte operations the searches need, with AVX2 (x86-64),
//! NEON (aarch64) and portable scalar implementations.

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(C, align(32))]
pub struct V32(pub [u8; 32]);

/// 16-byte vectors (a few operations, for node-parallel code).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(C, align(16))]
pub struct V16(pub [u8; 16]);

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod imp {
    use super::V32;
    use std::arch::x86_64::*;

    #[inline(always)]
    fn ld(a: &V32) -> __m256i {
        unsafe { _mm256_load_si256(a.0.as_ptr() as *const __m256i) }
    }

    #[inline(always)]
    fn st(v: __m256i) -> V32 {
        let mut o = V32([0; 32]);
        unsafe { _mm256_store_si256(o.0.as_mut_ptr() as *mut __m256i, v) };
        o
    }

    #[inline(always)]
    pub fn shuf(a: &V32, idx: &V32) -> V32 {
        unsafe { st(_mm256_shuffle_epi8(ld(a), ld(idx))) }
    }

    #[inline(always)]
    pub fn gather(a: &V32, idx: &V32) -> V32 {
        unsafe {
            let v = ld(a);
            let i = ld(idx);
            let lo = _mm256_permute2x128_si256(v, v, 0x00);
            let hi = _mm256_permute2x128_si256(v, v, 0x11);
            let x = _mm256_shuffle_epi8(lo, i);
            let y = _mm256_shuffle_epi8(hi, i);
            st(_mm256_blendv_epi8(x, y, _mm256_slli_epi16(i, 3)))
        }
    }

    #[inline(always)]
    pub fn movemask(a: &V32) -> u32 {
        unsafe { _mm256_movemask_epi8(ld(a)) as u32 }
    }

    #[inline(always)]
    pub fn is_zero(a: &V32) -> bool {
        unsafe {
            let v = ld(a);
            _mm256_testz_si256(v, v) != 0
        }
    }

    #[inline(always)]
    pub fn add(a: &V32, b: &V32) -> V32 {
        unsafe { st(_mm256_add_epi8(ld(a), ld(b))) }
    }

    #[inline(always)]
    pub fn sub(a: &V32, b: &V32) -> V32 {
        unsafe { st(_mm256_sub_epi8(ld(a), ld(b))) }
    }

    #[inline(always)]
    pub fn min(a: &V32, b: &V32) -> V32 {
        unsafe { st(_mm256_min_epu8(ld(a), ld(b))) }
    }

    #[inline(always)]
    pub fn xor(a: &V32, b: &V32) -> V32 {
        unsafe { st(_mm256_xor_si256(ld(a), ld(b))) }
    }

    #[inline(always)]
    pub fn and(a: &V32, b: &V32) -> V32 {
        unsafe { st(_mm256_and_si256(ld(a), ld(b))) }
    }

    #[inline(always)]
    pub fn or(a: &V32, b: &V32) -> V32 {
        unsafe { st(_mm256_or_si256(ld(a), ld(b))) }
    }

    /// Bytes 0..16 of `a` and `b`, interleaved.
    #[inline(always)]
    pub fn zip_lo(a: &V32, b: &V32) -> V32 {
        unsafe {
            // Lane 0: a0..7/b0..7, lane 1: a8..15/b8..15.
            let x = _mm256_permute4x64_epi64::<0b01_01_00_00>(ld(a));
            let y = _mm256_permute4x64_epi64::<0b01_01_00_00>(ld(b));
            st(_mm256_unpacklo_epi8(x, y))
        }
    }

    /// Bytes 16..32 of `a` and `b`, interleaved.
    #[inline(always)]
    pub fn zip_hi(a: &V32, b: &V32) -> V32 {
        unsafe {
            let x = _mm256_permute4x64_epi64::<0b11_11_10_10>(ld(a));
            let y = _mm256_permute4x64_epi64::<0b11_11_10_10>(ld(b));
            st(_mm256_unpacklo_epi8(x, y))
        }
    }

    /// Shift every byte left by N bits (bits leaving a byte may enter the next one; only bit
    /// 7 of the result is meaningful).
    #[inline(always)]
    pub fn shl<const N: i32>(a: &V32) -> V32 {
        unsafe { st(_mm256_slli_epi16::<N>(ld(a))) }
    }

    use super::V16;

    #[inline(always)]
    fn ld16(a: &V16) -> __m128i {
        unsafe { _mm_load_si128(a.0.as_ptr() as *const __m128i) }
    }

    #[inline(always)]
    fn st16(v: __m128i) -> V16 {
        let mut o = V16([0; 16]);
        unsafe { _mm_store_si128(o.0.as_mut_ptr() as *mut __m128i, v) };
        o
    }

    #[inline(always)]
    pub fn halves(a: &V32) -> (V16, V16) {
        unsafe {
            let v = ld(a);
            (
                st16(_mm256_castsi256_si128(v)),
                st16(_mm256_extracti128_si256::<1>(v)),
            )
        }
    }

    #[inline(always)]
    pub fn xor16(a: &V16, b: &V16) -> V16 {
        unsafe { st16(_mm_xor_si128(ld16(a), ld16(b))) }
    }

    #[inline(always)]
    pub fn and16(a: &V16, b: &V16) -> V16 {
        unsafe { st16(_mm_and_si128(ld16(a), ld16(b))) }
    }

    #[inline(always)]
    pub fn or16(a: &V16, b: &V16) -> V16 {
        unsafe { st16(_mm_or_si128(ld16(a), ld16(b))) }
    }

    /// Bit i set if byte i is not zero.
    #[inline(always)]
    pub fn nonzero16(a: &V16) -> u32 {
        unsafe {
            !(_mm_movemask_epi8(_mm_cmpeq_epi8(ld16(a), _mm_setzero_si128())) as u32) & 0xffff
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use super::V32;
    use std::arch::aarch64::*;

    #[inline(always)]
    fn ld(a: &V32) -> (uint8x16_t, uint8x16_t) {
        unsafe { (vld1q_u8(a.0.as_ptr()), vld1q_u8(a.0.as_ptr().add(16))) }
    }

    #[inline(always)]
    fn st(v: (uint8x16_t, uint8x16_t)) -> V32 {
        let mut o = V32([0; 32]);
        unsafe {
            vst1q_u8(o.0.as_mut_ptr(), v.0);
            vst1q_u8(o.0.as_mut_ptr().add(16), v.1);
        }
        o
    }

    #[inline(always)]
    fn map2(a: &V32, b: &V32, f: impl Fn(uint8x16_t, uint8x16_t) -> uint8x16_t) -> V32 {
        let (a0, a1) = ld(a);
        let (b0, b1) = ld(b);
        st((f(a0, b0), f(a1, b1)))
    }

    /// pshufb: per 16-byte lane, index bit 7 gives 0, else the low nibble selects.
    #[inline(always)]
    pub fn shuf(a: &V32, idx: &V32) -> V32 {
        unsafe {
            let m = vdupq_n_u8(0x8f);
            map2(a, idx, |t, i| vqtbl1q_u8(t, vandq_u8(i, m)))
        }
    }

    #[inline(always)]
    pub fn gather(a: &V32, idx: &V32) -> V32 {
        unsafe {
            let (a0, a1) = ld(a);
            let t = uint8x16x2_t(a0, a1);
            let (i0, i1) = ld(idx);
            st((vqtbl2q_u8(t, i0), vqtbl2q_u8(t, i1)))
        }
    }

    #[inline(always)]
    pub fn movemask(a: &V32) -> u32 {
        unsafe {
            const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let w = vld1q_u8(W.as_ptr());
            let (a0, a1) = ld(a);
            let m0 = vandq_u8(vcltzq_s8(vreinterpretq_s8_u8(a0)), w);
            let m1 = vandq_u8(vcltzq_s8(vreinterpretq_s8_u8(a1)), w);
            // Pairwise sums: bytes, then quads, then the four mask bytes in lane 0.
            let s = vpaddq_u8(m0, m1);
            let s = vpaddq_u8(s, s);
            let s = vpaddq_u8(s, s);
            vgetq_lane_u32::<0>(vreinterpretq_u32_u8(s))
        }
    }

    #[inline(always)]
    pub fn is_zero(a: &V32) -> bool {
        unsafe {
            let (a0, a1) = ld(a);
            vmaxvq_u8(vorrq_u8(a0, a1)) == 0
        }
    }

    #[inline(always)]
    pub fn add(a: &V32, b: &V32) -> V32 {
        unsafe { map2(a, b, |x, y| vaddq_u8(x, y)) }
    }

    #[inline(always)]
    pub fn sub(a: &V32, b: &V32) -> V32 {
        unsafe { map2(a, b, |x, y| vsubq_u8(x, y)) }
    }

    #[inline(always)]
    pub fn min(a: &V32, b: &V32) -> V32 {
        unsafe { map2(a, b, |x, y| vminq_u8(x, y)) }
    }

    #[inline(always)]
    pub fn xor(a: &V32, b: &V32) -> V32 {
        unsafe { map2(a, b, |x, y| veorq_u8(x, y)) }
    }

    #[inline(always)]
    pub fn and(a: &V32, b: &V32) -> V32 {
        unsafe { map2(a, b, |x, y| vandq_u8(x, y)) }
    }

    #[inline(always)]
    pub fn or(a: &V32, b: &V32) -> V32 {
        unsafe { map2(a, b, |x, y| vorrq_u8(x, y)) }
    }

    #[inline(always)]
    pub fn zip_lo(a: &V32, b: &V32) -> V32 {
        unsafe {
            let (a0, _) = ld(a);
            let (b0, _) = ld(b);
            st((vzip1q_u8(a0, b0), vzip2q_u8(a0, b0)))
        }
    }

    #[inline(always)]
    pub fn zip_hi(a: &V32, b: &V32) -> V32 {
        unsafe {
            let (_, a1) = ld(a);
            let (_, b1) = ld(b);
            st((vzip1q_u8(a1, b1), vzip2q_u8(a1, b1)))
        }
    }

    #[inline(always)]
    pub fn shl<const N: i32>(a: &V32) -> V32 {
        unsafe {
            let (a0, a1) = ld(a);
            st((vshlq_n_u8::<N>(a0), vshlq_n_u8::<N>(a1)))
        }
    }

    use super::V16;

    #[inline(always)]
    fn ld16(a: &V16) -> uint8x16_t {
        unsafe { vld1q_u8(a.0.as_ptr()) }
    }

    #[inline(always)]
    fn st16(v: uint8x16_t) -> V16 {
        let mut o = V16([0; 16]);
        unsafe { vst1q_u8(o.0.as_mut_ptr(), v) };
        o
    }

    #[inline(always)]
    pub fn halves(a: &V32) -> (V16, V16) {
        let (a0, a1) = ld(a);
        (st16(a0), st16(a1))
    }

    #[inline(always)]
    pub fn xor16(a: &V16, b: &V16) -> V16 {
        unsafe { st16(veorq_u8(ld16(a), ld16(b))) }
    }

    #[inline(always)]
    pub fn and16(a: &V16, b: &V16) -> V16 {
        unsafe { st16(vandq_u8(ld16(a), ld16(b))) }
    }

    #[inline(always)]
    pub fn or16(a: &V16, b: &V16) -> V16 {
        unsafe { st16(vorrq_u8(ld16(a), ld16(b))) }
    }

    #[inline(always)]
    pub fn nonzero16(a: &V16) -> u32 {
        unsafe {
            const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let a = ld16(a);
            let m = vandq_u8(vtstq_u8(a, a), vld1q_u8(W.as_ptr()));
            let m = vpaddq_u8(m, m);
            let m = vpaddq_u8(m, m);
            let m = vpaddq_u8(m, m);
            vgetq_lane_u16::<0>(vreinterpretq_u16_u8(m)) as u32
        }
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
mod imp {
    use super::V32;

    #[inline(always)]
    fn zip(a: &V32, b: &V32, f: impl Fn(u8, u8) -> u8) -> V32 {
        V32(std::array::from_fn(|i| f(a.0[i], b.0[i])))
    }

    #[inline(always)]
    pub fn shuf(a: &V32, idx: &V32) -> V32 {
        V32(std::array::from_fn(|i| {
            let x = idx.0[i];
            if x & 0x80 != 0 {
                0
            } else {
                a.0[(i & 16) + (x & 15) as usize]
            }
        }))
    }

    #[inline(always)]
    pub fn gather(a: &V32, idx: &V32) -> V32 {
        V32(std::array::from_fn(|i| a.0[(idx.0[i] & 31) as usize]))
    }

    #[inline(always)]
    pub fn movemask(a: &V32) -> u32 {
        (0..32).fold(0, |m, i| m | ((a.0[i] >> 7) as u32) << i)
    }

    #[inline(always)]
    pub fn is_zero(a: &V32) -> bool {
        a.0.iter().all(|&x| x == 0)
    }

    #[inline(always)]
    pub fn add(a: &V32, b: &V32) -> V32 {
        zip(a, b, u8::wrapping_add)
    }

    #[inline(always)]
    pub fn sub(a: &V32, b: &V32) -> V32 {
        zip(a, b, u8::wrapping_sub)
    }

    #[inline(always)]
    pub fn min(a: &V32, b: &V32) -> V32 {
        zip(a, b, u8::min)
    }

    #[inline(always)]
    pub fn xor(a: &V32, b: &V32) -> V32 {
        zip(a, b, |x, y| x ^ y)
    }

    #[inline(always)]
    pub fn and(a: &V32, b: &V32) -> V32 {
        zip(a, b, |x, y| x & y)
    }

    #[inline(always)]
    pub fn or(a: &V32, b: &V32) -> V32 {
        zip(a, b, |x, y| x | y)
    }

    #[inline(always)]
    pub fn zip_lo(a: &V32, b: &V32) -> V32 {
        V32(std::array::from_fn(|i| {
            if i & 1 == 0 { a.0[i / 2] } else { b.0[i / 2] }
        }))
    }

    #[inline(always)]
    pub fn zip_hi(a: &V32, b: &V32) -> V32 {
        V32(std::array::from_fn(|i| {
            if i & 1 == 0 {
                a.0[16 + i / 2]
            } else {
                b.0[16 + i / 2]
            }
        }))
    }

    #[inline(always)]
    pub fn shl<const N: i32>(a: &V32) -> V32 {
        V32(a.0.map(|x| x << N))
    }

    use super::V16;

    #[inline(always)]
    pub fn halves(a: &V32) -> (V16, V16) {
        (
            V16(std::array::from_fn(|i| a.0[i])),
            V16(std::array::from_fn(|i| a.0[16 + i])),
        )
    }

    #[inline(always)]
    pub fn xor16(a: &V16, b: &V16) -> V16 {
        V16(std::array::from_fn(|i| a.0[i] ^ b.0[i]))
    }

    #[inline(always)]
    pub fn and16(a: &V16, b: &V16) -> V16 {
        V16(std::array::from_fn(|i| a.0[i] & b.0[i]))
    }

    #[inline(always)]
    pub fn or16(a: &V16, b: &V16) -> V16 {
        V16(std::array::from_fn(|i| a.0[i] | b.0[i]))
    }

    #[inline(always)]
    pub fn nonzero16(a: &V16) -> u32 {
        (0..16).fold(0, |m, i| m | ((a.0[i] != 0) as u32) << i)
    }
}

pub use imp::{
    add, and, and16, gather, halves, is_zero, min, movemask, nonzero16, or, or16, shl, shuf, sub,
    xor, xor16, zip_hi, zip_lo,
};

/// The wing-major form of 16 rows (row i: byte w = item w of node i, w < 24): row e of the
/// result holds item 2e of node i in byte i and item 2e + 1 in byte 16 + i (e < 12), by
/// four rounds of interleaving row i with row i + 8.
#[inline(always)]
pub fn transpose16(m: &[V32; 16]) -> [V32; 12] {
    let mut m = *m;
    for _ in 0..3 {
        let mut o = [V32::ZERO; 16];
        for i in 0..8 {
            o[2 * i] = zip_lo(&m[i], &m[i + 8]);
            o[2 * i + 1] = zip_hi(&m[i], &m[i + 8]);
        }
        m = o;
    }
    let mut o = [V32::ZERO; 12];
    for i in 0..6 {
        o[2 * i] = zip_lo(&m[i], &m[i + 8]);
        o[2 * i + 1] = zip_hi(&m[i], &m[i + 8]);
    }
    o
}

/// Transpose a 32 x 32 byte matrix in place (`m[i].0[j]` becomes `m[j].0[i]`), by five
/// rounds of interleaving row i with row i + 16.
#[inline(always)]
pub fn transpose32(m: &mut [V32; 32]) {
    for _ in 0..5 {
        let mut o = [V32::ZERO; 32];
        for i in 0..16 {
            o[2 * i] = zip_lo(&m[i], &m[i + 16]);
            o[2 * i + 1] = zip_hi(&m[i], &m[i + 16]);
        }
        *m = o;
    }
}

impl V32 {
    pub const ZERO: V32 = V32([0; 32]);

    pub fn from_slice(s: &[u8]) -> V32 {
        let mut v = [0u8; 32];
        v[..s.len()].copy_from_slice(s);
        V32(v)
    }

    /// `out[i] = self[idx[i]]` for every i (indices < 32).
    #[inline(always)]
    pub fn gather(self, idx: &V32) -> V32 {
        gather(&self, idx)
    }

    /// Bitmask of the bytes whose value has bit `b` set.
    #[inline(always)]
    pub fn bit_mask(self, b: u32) -> u32 {
        match b {
            0 => movemask(&shl::<7>(&self)),
            1 => movemask(&shl::<6>(&self)),
            2 => movemask(&shl::<5>(&self)),
            3 => movemask(&shl::<4>(&self)),
            4 => movemask(&shl::<3>(&self)),
            5 => movemask(&shl::<2>(&self)),
            6 => movemask(&shl::<1>(&self)),
            _ => movemask(&self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_match_scalar() {
        let mut v = [0u8; 32];
        let mut idx = [0u8; 32];
        let mut sh = [0u8; 32];
        for i in 0..32 {
            v[i] = (i * 7 + 3) as u8 ^ ((i as u8) << 4);
            idx[i] = ((i * 13 + 5) % 32) as u8;
            sh[i] = if i % 5 == 0 {
                0x80
            } else {
                ((i * 11) as u8 & 0x3f) | ((i as u8 & 1) << 6)
            };
        }
        let (a, b) = (V32(v), V32(idx));
        let r = a.gather(&b);
        for i in 0..32 {
            assert_eq!(r.0[i], v[idx[i] as usize]);
        }
        let s = shuf(&a, &V32(sh));
        for i in 0..32 {
            let want = if sh[i] & 0x80 != 0 {
                0
            } else {
                v[(i & 16) + (sh[i] & 15) as usize]
            };
            assert_eq!(s.0[i], want, "shuf byte {i}");
        }
        let m = movemask(&a);
        for i in 0..32 {
            assert_eq!(m >> i & 1, (v[i] >> 7) as u32);
        }
        for k in 0..1000u32 {
            let x = V32(std::array::from_fn(|i| {
                (k.wrapping_mul(2654435761) >> (i % 24)) as u8 ^ (i as u8 * 37)
            }));
            let want = (0..32).fold(0, |a, i| a | ((x.0[i] >> 7) as u32) << i);
            assert_eq!(movemask(&x), want);
        }
        for bit in 0..8 {
            let bm = a.bit_mask(bit);
            for i in 0..32 {
                assert_eq!(bm >> i & 1, (v[i] >> bit & 1) as u32);
            }
        }
        let (zl, zh) = (zip_lo(&a, &b), zip_hi(&a, &b));
        let o = or(&a, &b);
        for i in 0..32 {
            assert_eq!(zl.0[i], if i % 2 == 0 { v[i / 2] } else { idx[i / 2] });
            assert_eq!(
                zh.0[i],
                if i % 2 == 0 {
                    v[16 + i / 2]
                } else {
                    idx[16 + i / 2]
                }
            );
            assert_eq!(o.0[i], v[i] | idx[i]);
        }
        let mut m: [V32; 32] =
            std::array::from_fn(|i| V32(std::array::from_fn(|j| (i * 32 + j) as u8)));
        transpose32(&mut m);
        for i in 0..32 {
            for j in 0..32 {
                assert_eq!(m[i].0[j], (j * 32 + i) as u8, "transpose {i} {j}");
            }
        }
        let rows: [V32; 16] =
            std::array::from_fn(|i| V32(std::array::from_fn(|j| (i * 32 + j) as u8)));
        let t16 = transpose16(&rows);
        for e in 0..12 {
            for i in 0..16 {
                assert_eq!(t16[e].0[i], rows[i].0[2 * e], "transpose16 {e} {i}");
                assert_eq!(
                    t16[e].0[16 + i],
                    rows[i].0[2 * e + 1],
                    "transpose16 {e} {i}"
                );
            }
        }
        let (h0, h1) = halves(&a);
        let (g0, g1) = halves(&b);
        let (x16, a16, o16) = (xor16(&h0, &g0), and16(&h1, &g1), or16(&h0, &g1));
        for i in 0..16 {
            assert_eq!(x16.0[i], v[i] ^ idx[i]);
            assert_eq!(a16.0[i], v[16 + i] & idx[16 + i]);
            assert_eq!(o16.0[i], v[i] | idx[16 + i]);
        }
        let z = V16(std::array::from_fn(|i| {
            if i % 3 == 0 { 0 } else { 1 << (i % 8) }
        }));
        assert_eq!(
            nonzero16(&z),
            (0..16).filter(|i| i % 3 != 0).fold(0, |m, i| m | 1 << i)
        );
        assert!(is_zero(&V32::ZERO));
        for i in 0..32 {
            let mut z = V32::ZERO;
            z.0[i] = 1 << (i % 8);
            assert!(!is_zero(&z));
        }
        let mn = min(&a, &b);
        let ad = add(&a, &b);
        let sb = sub(&a, &b);
        for i in 0..32 {
            assert_eq!(mn.0[i], v[i].min(idx[i]));
            assert_eq!(ad.0[i], v[i].wrapping_add(idx[i]));
            assert_eq!(sb.0[i], v[i].wrapping_sub(idx[i]));
        }
    }
}
