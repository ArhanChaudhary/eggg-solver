//! For random wing permutations: the share that is pairable (with even wing parity, as at a
//! phase-1 endpoint) after some sequence of at most k moves, per frame and in any frame,
//! ignoring the centres. This bounds what a wing-only lookahead filter (any table over the
//! wing state) can prune k moves before the end of phase 1. Env: N (samples), SEED.
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES, WIDE_QUARTER};
use eggg_solver::phase1::Frame;
use eggg_solver::simd::V32;
use rand::SeedableRng;
use rand::seq::SliceRandom;

fn main() {
    let n: usize = std::env::var("N")
        .ok()
        .map_or(20000, |v| v.parse().unwrap());
    let seed: u64 = std::env::var("SEED").ok().map_or(1, |v| v.parse().unwrap());
    let frames = [Frame::new(0), Frame::new(1), Frame::new(2)];
    let dst: Vec<V32> = (0..N_MOVES)
        .map(|m| {
            let mut d = V32::ZERO;
            for t in 0..24 {
                d.0[MOVE_PERMS[m].wing_src[t] as usize] = t as u8;
            }
            d
        })
        .collect();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    // [k][frame or 3 = any]
    let mut hit = [[0usize; 4]; 4];
    let ok = |pos: &V32, odd: bool| -> [bool; 3] {
        std::array::from_fn(|f| !odd && frames[f].wings_ok(pos))
    };
    for _ in 0..n {
        let mut p: Vec<u8> = (0..24).collect();
        p.shuffle(&mut rng);
        let mut pos = V32::ZERO;
        pos.0[..24].copy_from_slice(&p);
        let odd = Cube::parity(&p) == 1;
        let mut best = [[false; 3]; 4];
        best[0] = ok(&pos, odd);
        for m1 in 0..N_MOVES {
            let p1 = dst[m1].gather(&pos);
            let o1 = odd ^ (WIDE_QUARTER >> m1 & 1 == 1);
            let r1 = ok(&p1, o1);
            for f in 0..3 {
                best[1][f] |= r1[f];
            }
            let mut a = ALLOWED_AFTER[m1];
            while a != 0 {
                let m2 = a.trailing_zeros() as usize;
                a &= a - 1;
                let p2 = dst[m2].gather(&p1);
                let o2 = o1 ^ (WIDE_QUARTER >> m2 & 1 == 1);
                let r2 = ok(&p2, o2);
                for f in 0..3 {
                    best[2][f] |= r2[f];
                }
                let mut b = ALLOWED_AFTER[m2];
                while b != 0 {
                    let m3 = b.trailing_zeros() as usize;
                    b &= b - 1;
                    let p3 = dst[m3].gather(&p2);
                    let o3 = o2 ^ (WIDE_QUARTER >> m3 & 1 == 1);
                    let r3 = ok(&p3, o3);
                    for f in 0..3 {
                        best[3][f] |= r3[f];
                    }
                }
            }
        }
        for k in 0..4 {
            let mut any = false;
            for f in 0..3 {
                // within k: at most k moves
                let w = (0..=k).any(|j| best[j][f]);
                if w {
                    hit[k][f] += 1;
                }
                any |= w;
            }
            if any {
                hit[k][3] += 1;
            }
        }
    }
    for (k, h) in hit.iter().enumerate() {
        println!(
            "within {k} moves: frame0 {:.4}% frame1 {:.4}% frame2 {:.4}% any {:.4}%",
            100.0 * h[0] as f64 / n as f64,
            100.0 * h[1] as f64 / n as f64,
            100.0 * h[2] as f64 / n as f64,
            100.0 * h[3] as f64 / n as f64
        );
    }
}
