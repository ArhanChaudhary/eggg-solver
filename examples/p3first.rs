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
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(11);
    for depth in [20u8, 19, 18] {
        let mut tot = 0u64;
        let mut found = 0;
        let t0 = Instant::now();
        for _ in 0..10 {
            let mut x = Cube3::SOLVED;
            for _ in 0..60 {
                x = x.apply(rng.gen_range(0..18));
            }
            let mut s = Search3::new(&t);
            s.node_limit = 50_000_000;
            if let Some(sol) = s.search_depth(&x, depth, eggg_solver::moves::N_MOVES) {
                assert_eq!(x.apply_moves(&sol), Cube3::SOLVED);
                found += 1;
            }
            tot += s.nodes;
        }
        println!(
            "depth {depth}: found {found}/10, mean nodes {:.3e}, {:.3} s total",
            tot as f64 / 10.0,
            t0.elapsed().as_secs_f64()
        );
    }
}
