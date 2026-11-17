//! Deterministic phase-3 measurement: bounded searches at fixed depths on fixed cubes.
use eggg_solver::cube3::Cube3;
use eggg_solver::phase3::{P3Tables, Search3};
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let k: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(6);
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), k, 8, false);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(77);
    let cubes: Vec<Cube3> = (0..12)
        .map(|_| {
            let mut x = Cube3::SOLVED;
            for _ in 0..60 {
                x = x.apply(rng.gen_range(0..18));
            }
            x
        })
        .collect();
    let t0 = Instant::now();
    let mut total = 0u64;
    let mut found = 0;
    for d in [14u8, 15, 16] {
        let mut n = 0u64;
        for x in &cubes {
            let mut s = Search3::new(&t);
            if let Some(sol) = s.search_depth(x, d, eggg_solver::moves::N_MOVES) {
                assert_eq!(x.apply_moves(&sol), Cube3::SOLVED);
                found += 1;
            }
            n += s.nodes;
        }
        println!("depth {d}: {n} nodes");
        total += n;
    }
    println!(
        "total {total} nodes, {found} found, {:.2} s",
        t0.elapsed().as_secs_f64()
    );
}
