//! Deterministic phase-1 check: endpoint-set hash and node counts on fixed cubes.
//! EGGG_P1_BIG: the large staging table; EGGG_P1_MASK=small|big: the move-mask table (the
//! large one with the large staging table). CUBES, EXTRA: the workload (default 5 cubes,
//! lengths d0..=d0+3).
//! The hash covers the endpoints in the order the search reports them.
use eggg_solver::cube::Cube;
use eggg_solver::moves::N_MOVES;
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::solver::{load_mask, load_p1};
use rand::SeedableRng;
use std::hash::{Hash, Hasher};
use std::time::Instant;

#[path = "common/perfc.rs"]
mod perf;

fn main() {
    let mask = std::env::var("EGGG_P1_MASK").ok();
    let big = std::env::var_os("EGGG_P1_BIG").is_some() || mask.as_deref() == Some("big");
    let env = |k: &str, d: u8| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    let (cubes, extra) = (env("CUBES", 5), env("EXTRA", 3));
    let dir = std::path::Path::new("tables");
    let mut p1 = load_p1(dir, big, 8, false);
    p1.table.data.collapse();
    if let Some(kind) = &mask {
        load_mask(&mut p1, dir, kind == "big", true);
        p1.mask.as_ref().unwrap().collapse();
    }
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let (mut nodes, mut reads, mut eps) = (0u64, 0u64, 0usize);
    let mut prof = [0u64; 48];
    let (ins, cyc) = (
        perf::Counter::new("instructions"),
        perf::Counter::new("cycles"),
    );
    if let (Some(i), Some(c)) = (&ins, &cyc) {
        i.start();
        c.start();
    }
    let t0 = Instant::now();
    for _ in 0..cubes {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        for len in d0..=d0 + extra {
            let mut found = vec![];
            let mut s = Search1::new(&p1, |p: &[u8], f| {
                found.push((p.to_vec(), f));
                true
            });
            s.dfs(&root, d0, len, N_MOVES);
            nodes += s.nodes;
            reads += s.reads;
            for (a, b) in prof.iter_mut().zip(s.prof) {
                *a += b;
            }
            eps += found.len();
            found.hash(&mut h);
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let (ni, nc) = match (&ins, &cyc) {
        (Some(i), Some(c)) => (i.stop(), c.stop()),
        _ => (0, 0),
    };
    println!(
        "endpoints {eps}, hash {:016x}, nodes {nodes}, reads {reads}, {dt:.3} s",
        h.finish(),
    );
    if ni > 0 {
        println!(
            "user instructions {ni} ({:.2}/node), cycles {nc}, IPC {:.2}",
            ni as f64 / nodes as f64,
            ni as f64 / nc.max(1) as f64
        );
    }
    #[cfg(feature = "p1prof")]
    if let Ok(path) = std::env::var("EGGG_W2_FILE") {
        // The captured lookahead inputs: 24 slot bytes, then odd | tight << 1.
        let cap = eggg_solver::wing2::take_capture();
        let mut out = Vec::with_capacity(cap.len() * 25);
        for (pos, odd, tight) in &cap {
            out.extend_from_slice(&pos.0[..24]);
            out.push(*odd as u8 | (*tight as u8) << 1);
        }
        std::fs::write(&path, out).unwrap();
        println!("captured {} lookahead tests to {path}", cap.len());
    }
    if prof.iter().any(|&x| x != 0) {
        for (n, v) in eggg_solver::phase1::PROF_NAMES.iter().zip(prof) {
            if !n.is_empty() {
                println!("{n:>20} {v:>14}");
            }
        }
    }
}
