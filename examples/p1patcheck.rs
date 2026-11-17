//! Deterministic check of the per-pair phase-1 searches (pattern table) on p1check's cubes and
//! lengths: the union of the three pair searches' (endpoint, frame) pairs, its hash, nodes and
//! time, and the same for the plain search (one tree for all pairs, staging table); with CHECK
//! set, the two sets must be equal.
//! EGGG_P1_BIG: the large staging table; EGGG_P1_PAT=big|small|none: the pattern table (default
//! small; none runs the plain search only); EGGG_P1_MASK=big|small|virtual (with EGGG_P1_PAT=none):
//! the plain search with that move-mask table; THREADS=n: n copies at once (wall time, so
//! the throughput with n threads sharing the memory system).
use eggg_solver::cube::Cube;
use eggg_solver::moves::N_MOVES;
use eggg_solver::phase1::{ALL_PAIRS, P1State, P1Tables, Search1};
use eggg_solver::solver::{load_mask, load_p1, load_pat};
use rand::SeedableRng;
use std::hash::{Hash, Hasher};
use std::time::Instant;

type Eps = Vec<(Vec<u8>, u8)>;

/// The (endpoint, frame) pairs per (cube, length), sorted, and the nodes; by the pair
/// searches (`pairs`) or the plain one.
fn workload(p1: &P1Tables, pairs: bool) -> (Vec<Eps>, u64, u64) {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
    let mut out = vec![];
    let mut nodes = 0;
    let mut reads = 0u64;
    let env = |k: &str, d: u8| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    let (cubes, from, extra) = (env("CUBES", 5), env("FROM", 0), env("EXTRA", 3));
    for _ in 0..cubes {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        let searches: Vec<(usize, u8)> = if pairs {
            (0..3).map(|c| (c, p1.distance_pair(&root, c))).collect()
        } else {
            vec![(ALL_PAIRS, d0)]
        };
        for len in d0 + from..=d0 + extra {
            let mut found: Eps = vec![];
            for &(c, d) in &searches {
                if d > len {
                    continue;
                }
                let mut s = Search1::new(p1, |p: &[u8], f| {
                    for g in 0..3 {
                        if f >> g & 1 == 1 {
                            found.push((p.to_vec(), g));
                        }
                    }
                    true
                });
                s.pair = c;
                s.dfs(&root, d, len, N_MOVES);
                nodes += s.nodes;
                reads += s.reads;
            }
            found.sort();
            out.push(found);
        }
    }
    (out, nodes, reads)
}

fn run(p1: &P1Tables, pairs: bool, threads: usize) -> (Vec<Eps>, (u64, u64), f64) {
    let t0 = Instant::now();
    let res: Vec<(Vec<Eps>, u64, u64)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|_| s.spawn(|| workload(p1, pairs)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let el = t0.elapsed().as_secs_f64();
    let (eps, nodes, reads) = res.into_iter().next().unwrap();
    (eps, (nodes, reads), el)
}

fn report(name: &str, eps: &[Eps], (nodes, reads): (u64, u64), el: f64, threads: usize) {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    eps.hash(&mut h);
    let n: usize = eps.iter().map(|e| e.len()).sum();
    println!(
        "{name}: (endpoint, frame) pairs {n}, hash {:016x}, nodes {nodes}, table reads {reads}, {el:.2} s wall \
         ({threads} copies: {:.1} M nodes/s in all)",
        h.finish(),
        nodes as f64 * threads as f64 / el / 1e6
    );
}

fn main() {
    // Table generation: at most 8 threads (a shared machine).
    eggg_solver::par::set_threads(8);
    let big = std::env::var_os("EGGG_P1_BIG").is_some();
    let pat = std::env::var("EGGG_P1_PAT").unwrap_or_else(|_| "small".into());
    let mask = std::env::var_os("EGGG_P1_MASK").is_some();
    let check = std::env::var_os("CHECK").is_some();
    let threads: usize = std::env::var("THREADS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1);
    let dir = std::path::Path::new("tables");
    let mut p1 = load_p1(dir, big || mask, 8, false);
    p1.table.data.collapse();
    if pat != "none" {
        let t0 = Instant::now();
        load_pat(&mut p1, dir, pat == "big", 8, true);
        p1.pat.as_ref().unwrap().table.data.collapse();
        match std::env::var("EGGG_P1_MASK").as_deref() {
            // Per-pair searches with batches of pattern-decreasing moves: computed from the
            // pattern table (virtual: directly; lazy: through the pair table's indexing), or
            // the pair mask table (pair).
            Ok("virtual") => p1.mask = Some(eggg_solver::p1mask::MaskTable::virtual_table()),
            Ok("lazy") => p1.mask = Some(eggg_solver::p1mask::MaskTable::lazy_pair_table()),
            Ok("pair") => load_mask(&mut p1, dir, false, true),
            _ => {}
        }
        eprintln!(
            "pattern table loaded in {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        let (eps, nodes, el) = run(&p1, true, threads);
        report("pair searches", &eps, nodes, el, threads);
        if check {
            p1.mask = None;
            let (want, nodes0, el0) = run(&p1, false, threads);
            report("plain search", &want, nodes0, el0, threads);
            assert!(eps == want, "different endpoint sets");
            println!("same (endpoint, frame) sets");
        }
    } else if mask {
        let t0 = Instant::now();
        let kind = std::env::var("EGGG_P1_MASK").unwrap();
        if kind == "virtual" {
            p1.mask = Some(eggg_solver::p1mask::MaskTable::virtual_table());
        } else {
            load_mask(&mut p1, dir, kind == "big", true);
        }
        p1.mask.as_ref().unwrap().collapse();
        eprintln!("mask table loaded in {:.1} s", t0.elapsed().as_secs_f64());
        let (eps, nodes, el) = run(&p1, false, threads);
        report("masked search", &eps, nodes, el, threads);
        if check {
            p1.mask = None;
            let (want, nodes0, el0) = run(&p1, false, threads);
            report("plain search", &want, nodes0, el0, threads);
            assert!(eps == want, "different endpoint sets");
            println!("same (endpoint, frame) sets");
        }
    } else {
        let (eps, nodes, el) = run(&p1, false, threads);
        report("plain search", &eps, nodes, el, threads);
    }
}
