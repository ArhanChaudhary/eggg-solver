//! Combinatorial ranking helpers (combinations and permutations).

/// Binomial coefficients C(n, k) for n, k < 32.
pub const BINOM: [[u32; 32]; 32] = {
    let mut t = [[0u32; 32]; 32];
    let mut n = 0;
    while n < 32 {
        t[n][0] = 1;
        let mut k = 1;
        while k <= n {
            t[n][k] = t[n - 1][k - 1] + if k < n { t[n - 1][k] } else { 0 };
            k += 1;
        }
        n += 1;
    }
    t
};

pub const fn binom(n: usize, k: usize) -> u32 {
    if k > n { 0 } else { BINOM[n][k] }
}

/// Rank of a k-subset of {0..n} given as a bitmask (colex order).
#[inline]
pub fn comb_rank(mut mask: u32) -> u32 {
    let mut r = 0;
    let mut k = 1;
    while mask != 0 {
        let i = mask.trailing_zeros() as usize;
        r += BINOM[i][k];
        k += 1;
        mask &= mask - 1;
    }
    r
}

/// Inverse of [`comb_rank`]: the k-subset (bitmask) of rank `r` among subsets of {0..n}.
pub fn comb_unrank(mut r: u32, n: usize, k: usize) -> u32 {
    let mut mask = 0;
    let mut k = k;
    let mut i = n;
    while k > 0 {
        i -= 1;
        if BINOM[i][k] <= r {
            r -= BINOM[i][k];
            mask |= 1 << i;
            k -= 1;
        }
    }
    mask
}

pub const FACT: [u64; 21] = {
    let mut t = [1u64; 21];
    let mut i = 1;
    while i < 21 {
        t[i] = t[i - 1] * i as u64;
        i += 1;
    }
    t
};

/// Lehmer rank of a permutation of {0..n} (n <= 20).
pub fn perm_rank(p: &[u8]) -> u64 {
    let n = p.len();
    let mut r = 0u64;
    let mut used = 0u32;
    for i in 0..n {
        let x = p[i] as u32;
        let smaller = (!used & ((1 << x) - 1)).count_ones() as u64;
        r += smaller * FACT[n - 1 - i];
        used |= 1 << x;
    }
    r
}

pub fn perm_unrank(mut r: u64, n: usize) -> Vec<u8> {
    let mut avail: Vec<u8> = (0..n as u8).collect();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let f = FACT[n - 1 - i];
        let q = (r / f) as usize;
        r %= f;
        out.push(avail.remove(q));
    }
    out
}

/// Parity (0 even, 1 odd) of a permutation of {0..n}.
pub fn perm_parity(p: &[u8]) -> u8 {
    let mut par = 0u8;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            par ^= (p[i] > p[j]) as u8;
        }
    }
    par
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comb_roundtrip() {
        for n in [8usize, 12, 16, 24] {
            for k in [1usize, 4, 8] {
                if k > n {
                    continue;
                }
                let total = binom(n, k);
                for r in (0..total).step_by((total as usize / 1000).max(1)) {
                    let m = comb_unrank(r, n, k);
                    assert_eq!(m.count_ones() as usize, k);
                    assert!(m < (1 << n));
                    assert_eq!(comb_rank(m), r);
                }
            }
        }
    }

    #[test]
    fn perm_roundtrip() {
        for r in (0..FACT[8]).step_by(97) {
            let p = perm_unrank(r, 8);
            assert_eq!(perm_rank(&p), r);
        }
    }
}
