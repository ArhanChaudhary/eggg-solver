//! Time the big phase-2 table's per-entry generator work (decode and 17 canonical
//! neighbours), and its search index, without the table data.
use eggg_solver::p2big::Big;
use eggg_solver::solver::load_tables_with;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let t = load_tables_with(std::path::Path::new("tables"), 0, false, false, 4, false);
    let t0 = Instant::now();
    let b = Big::new(&t.p2);
    println!(
        "setup {:.2} s: {} classes, {:.2} GB",
        t0.elapsed().as_secs_f64(),
        b.classes(),
        b.bytes() as f64 / 1e9
    );
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
    let idx: Vec<usize> = (0..200_000).map(|_| rng.gen_range(0..b.n)).collect();
    let t0 = Instant::now();
    let mut x = 0;
    for &i in &idx {
        x ^= b.probe(&t.p2, i);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{:.0} ns per entry expansion ({x:x}); 1.5e11 expansions on 48 threads: {:.0} min",
        dt / idx.len() as f64 * 1e9,
        dt / idx.len() as f64 * 1.5e11 / 48.0 / 60.0
    );
}
