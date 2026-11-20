//! Multi-threaded phase-1 throughput: THREADS threads (default 8), each searching CUBES
//! random cubes (default 4, seeded by thread) at lengths d0..=d0+EXTRA (default 4), as in
//! p1check, with one shared table load (the move-mask table, EGGG_P1_MASK=small). Prints
//! the endpoint hash (per thread, combined), nodes, wall time and nodes per second.
use eggg_solver::cube::Cube;
use eggg_solver::moves::N_MOVES;
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::solver::{load_mask, load_p1};
use rand::SeedableRng;
use std::hash::{Hash, Hasher};
use std::time::Instant;

fn main() {
    let env = |k: &str, d: usize| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    let (threads, cubes, extra) = (env("THREADS", 8), env("CUBES", 4), env("EXTRA", 4) as u8);
    let dir = std::path::Path::new("tables");
    let mut p1 = load_p1(dir, false, 8, false);
    p1.table.data.collapse();
    load_mask(&mut p1, dir, false, true);
    p1.mask.as_ref().unwrap().collapse();
    let p1 = &p1;
    let t0 = Instant::now();
    let res: Vec<(u64, u64, usize, f64)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|k| {
                s.spawn(move || {
                    let t1 = Instant::now();
                    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(100 + k as u64);
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    let (mut nodes, mut eps) = (0u64, 0usize);
                    for _ in 0..cubes {
                        let cube = Cube::random(&mut rng);
                        let root = P1State::from_cube(&cube);
                        let d0 = p1.distance(&root);
                        for len in d0..=d0 + extra {
                            let mut found = vec![];
                            let mut s = Search1::new(p1, |p: &[u8], f| {
                                found.push((p.to_vec(), f));
                                true
                            });
                            s.dfs(&root, d0, len, N_MOVES);
                            nodes += s.nodes;
                            eps += found.len();
                            found.hash(&mut h);
                        }
                    }
                    (h.finish(), nodes, eps, t1.elapsed().as_secs_f64())
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wall = t0.elapsed().as_secs_f64();
    let hash = res.iter().fold(0u64, |a, r| a.rotate_left(7) ^ r.0);
    let nodes: u64 = res.iter().map(|r| r.1).sum();
    let eps: usize = res.iter().map(|r| r.2).sum();
    let busy: f64 = res.iter().map(|r| r.3).sum();
    println!(
        "threads {threads}: endpoints {eps}, hash {hash:016x}, nodes {nodes}, wall {wall:.3} s, \
         thread-time {busy:.3} s, {:.1} Mnodes/s per thread-second",
        nodes as f64 / busy / 1e6
    );
}
