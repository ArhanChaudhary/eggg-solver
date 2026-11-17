//! Deterministic phase-1 check: endpoint-set hash and node counts on fixed cubes.
use eggg_solver::cube::Cube;
use eggg_solver::moves::N_MOVES;
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::solver::load_p1;
use rand::SeedableRng;
use std::hash::{Hash, Hasher};
use std::time::Instant;

fn main() {
    // EGGG_P1_BIG set: the large staging table.
    let big = std::env::var_os("EGGG_P1_BIG").is_some();
    let p1 = load_p1(std::path::Path::new("tables"), big, 8, false);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let (mut nodes, mut eps) = (0u64, 0usize);
    let t0 = Instant::now();
    for _ in 0..5 {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        for len in d0..=d0 + 3 {
            let mut found = vec![];
            let mut s = Search1::new(&p1, |p: &[u8], f| {
                found.push((p.to_vec(), f));
                true
            });
            s.dfs(&root, d0, len, N_MOVES);
            nodes += s.nodes;
            eps += found.len();
            found.hash(&mut h);
        }
    }
    println!(
        "endpoints {eps}, hash {:016x}, nodes {nodes}, {:.2} s",
        h.finish(),
        t0.elapsed().as_secs_f64()
    );
}
