//! How many distinct low-slot preimages a 3-move lookahead (last move a wide quarter turn,
//! endpoint parity even) has per frame, and its pass rate on random wing states.
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES, WIDE_QUARTER};
use eggg_solver::phase1::Frame;
use eggg_solver::simd::V32;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use std::collections::HashSet;

fn main() {
    let frames = [Frame::new(0), Frame::new(1), Frame::new(2)];
    let dst: Vec<[u8; 24]> = (0..N_MOVES)
        .map(|m| {
            let mut d = [0u8; 24];
            for t in 0..24 {
                d[MOVE_PERMS[m].wing_src[t] as usize] = t as u8;
            }
            d
        })
        .collect();
    // [parity][frame]: distinct preimages (24-bit masks).
    let mut pre: Vec<Vec<HashSet<u32>>> = vec![vec![HashSet::new(); 3]; 2];
    for m1 in 0..N_MOVES {
        let mut a = ALLOWED_AFTER[m1];
        while a != 0 {
            let m2 = a.trailing_zeros() as usize;
            a &= a - 1;
            let mut b = ALLOWED_AFTER[m2] & WIDE_QUARTER;
            while b != 0 {
                let m3 = b.trailing_zeros() as usize;
                b &= b - 1;
                let flips = (WIDE_QUARTER >> m1 & 1) ^ (WIDE_QUARTER >> m2 & 1) ^ 1;
                let par = flips as usize; // node parity must equal the flips
                for f in 0..3 {
                    let mut p = 0u32;
                    for x in 0..24 {
                        let y = dst[m3][dst[m2][dst[m1][x] as usize] as usize];
                        if frames[f].low_slots.contains(&y) {
                            p |= 1 << x;
                        }
                    }
                    pre[par][f].insert(p);
                }
            }
        }
    }
    for par in 0..2 {
        println!(
            "parity {par}: preimages per frame {:?}",
            pre[par].iter().map(|s| s.len()).collect::<Vec<_>>()
        );
    }
    // Pass rate on random states (pairable w.r.t. some preimage, even high count).
    let n: usize = std::env::var("N").ok().map_or(3000, |v| v.parse().unwrap());
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
    let lists: Vec<Vec<Vec<u32>>> = pre
        .iter()
        .map(|v| v.iter().map(|s| s.iter().copied().collect()).collect())
        .collect();
    let mut pass = 0;
    for _ in 0..n {
        let mut p: Vec<u8> = (0..24).collect();
        p.shuffle(&mut rng);
        let odd = Cube::parity(&p) == 1;
        let mut ok = false;
        'f: for f in 0..3 {
            for &pm in &lists[odd as usize][f] {
                let mut good = true;
                let mut high = 0;
                for e in 0..12 {
                    let (a, b) = (p[2 * e] as u32, p[2 * e + 1] as u32);
                    if (pm >> a & 1) == (pm >> b & 1) {
                        good = false;
                        break;
                    }
                    // the high piece of the pair in the preimage
                    let hp = if frames[f].high_pieces >> (2 * e) & 1 == 1 {
                        2 * e
                    } else {
                        2 * e + 1
                    };
                    high ^= pm >> p[hp] & 1;
                }
                if good && high == 0 {
                    ok = true;
                    break 'f;
                }
            }
        }
        pass += ok as usize;
    }
    let _ = V32::ZERO;
    println!("random pass rate {:.2}%", 100.0 * pass as f64 / n as f64);
}
