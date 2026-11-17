use eggg_solver::cube3::Cube3;
use eggg_solver::phase3::{P3Tables, Search3};
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let k: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(2);
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), k, 8, true);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
    // Distribution of h48 values over random cubes.
    let mut hist = [0u64; 20];
    let mut chist = [0u64; 20];
    for _ in 0..100000 {
        let mut x = Cube3::SOLVED;
        for _ in 0..60 {
            x = x.apply(rng.gen_range(0..18));
        }
        hist[t.h48_bound(&x) as usize] += 1;
        let s = Search3::new(&t);
        chist[s.lower_bound(&x) as usize] += 1;
    }
    println!("h48 value histogram: {:?}", &hist[..16]);
    println!("full bound histogram: {:?}", &chist[..16]);
    // Short scrambles: optimal lengths must not exceed the scramble length.
    for len in [8usize, 10, 12, 13] {
        for _ in 0..20 {
            let mut x = Cube3::SOLVED;
            for _ in 0..len {
                x = x.apply(rng.gen_range(0..18));
            }
            let mut s = Search3::new(&t);
            let sol = s.solve(&x, 20).expect("no solution");
            assert_eq!(x.apply_moves(&sol), Cube3::SOLVED);
            assert!(sol.len() <= len);
        }
    }
    println!("short scrambles ok");
    let mut x = Cube3::SOLVED;
    for _ in 0..60 {
        x = x.apply(rng.gen_range(0..18));
    }
    let mut s = Search3::new(&t);
    let h = s.lower_bound(&x);
    for d in h..=20 {
        let t0 = Instant::now();
        let n0 = s.nodes;
        let r = s.search_depth(&x, d, eggg_solver::moves::N_MOVES);
        println!(
            "depth {d}: {} nodes, {:.3} s, found {}",
            s.nodes - n0,
            t0.elapsed().as_secs_f64(),
            r.is_some()
        );
        if let Some(sol) = &r {
            assert_eq!(x.apply_moves(sol), Cube3::SOLVED, "bad solution");
            println!("verified {} moves", sol.len());
        }
        if r.is_some() || t0.elapsed().as_secs_f64() > 60.0 {
            break;
        }
    }
}
