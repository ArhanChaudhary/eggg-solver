//! A 32-byte vector with an arbitrary byte gather (AVX2 when available, scalar otherwise).

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(C, align(32))]
pub struct V32(pub [u8; 32]);

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
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use std::arch::x86_64::*;
            let v = _mm256_load_si256(self.0.as_ptr() as *const __m256i);
            let i = _mm256_load_si256(idx.0.as_ptr() as *const __m256i);
            let lo = _mm256_permute2x128_si256(v, v, 0x00);
            let hi = _mm256_permute2x128_si256(v, v, 0x11);
            let a = _mm256_shuffle_epi8(lo, i);
            let b = _mm256_shuffle_epi8(hi, i);
            let sel = _mm256_slli_epi16(i, 3); // bit 4 -> bit 7
            let r = _mm256_blendv_epi8(a, b, sel);
            let mut out = V32::ZERO;
            _mm256_store_si256(out.0.as_mut_ptr() as *mut __m256i, r);
            out
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut out = [0u8; 32];
            for i in 0..32 {
                out[i] = self.0[(idx.0[i] & 31) as usize];
            }
            V32(out)
        }
    }

    /// Bitmask of the bytes equal to `x` (bit i for byte i).
    #[inline(always)]
    pub fn eq_mask(self, x: u8) -> u32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use std::arch::x86_64::*;
            let v = _mm256_load_si256(self.0.as_ptr() as *const __m256i);
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(x as i8))) as u32
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut m = 0;
            for i in 0..32 {
                m |= ((self.0[i] == x) as u32) << i;
            }
            m
        }
    }

    /// Bitmask of the bytes whose value has bit `b` set.
    #[inline(always)]
    pub fn bit_mask(self, b: u32) -> u32 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use std::arch::x86_64::*;
            let v = _mm256_load_si256(self.0.as_ptr() as *const __m256i);
            let shifted = match b {
                0 => _mm256_slli_epi16(v, 7),
                1 => _mm256_slli_epi16(v, 6),
                2 => _mm256_slli_epi16(v, 5),
                3 => _mm256_slli_epi16(v, 4),
                4 => _mm256_slli_epi16(v, 3),
                5 => _mm256_slli_epi16(v, 2),
                6 => _mm256_slli_epi16(v, 1),
                _ => v,
            };
            _mm256_movemask_epi8(shifted) as u32
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut m = 0;
            for i in 0..32 {
                m |= (((self.0[i] >> b) & 1) as u32) << i;
            }
            m
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_matches_scalar() {
        let mut v = [0u8; 32];
        let mut idx = [0u8; 32];
        for i in 0..32 {
            v[i] = (i * 7 + 3) as u8;
            idx[i] = ((i * 13 + 5) % 32) as u8;
        }
        let r = V32(v).gather(&V32(idx));
        for i in 0..32 {
            assert_eq!(r.0[i], v[idx[i] as usize]);
        }
        assert_eq!(V32(v).eq_mask(3), 1);
        assert_eq!(V32(v).bit_mask(0).count_ones(), 16);
    }
}
