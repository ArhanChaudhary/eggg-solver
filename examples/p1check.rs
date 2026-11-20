//! Deterministic phase-1 check: endpoint-set hash and node counts on fixed cubes.
//! EGGG_P1_BIG: the large staging table; EGGG_P1_MASK=small|wide|big: the move-mask table (the
//! large one with the large staging table). CUBES, EXTRA: the workload (default 5 cubes,
//! lengths d0..=d0+3).
//! The hash covers the endpoints in the order the search reports them.
use eggg_solver::cube::Cube;
use eggg_solver::moves::N_MOVES;
use eggg_solver::p1mask::MaskSel;
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::solver::{load_mask_sel, load_p1};
use rand::SeedableRng;
use std::hash::{Hash, Hasher};
use std::time::Instant;

fn main() {
    let mask = std::env::var("EGGG_P1_MASK").ok();
    let big = std::env::var_os("EGGG_P1_BIG").is_some() || mask.as_deref() == Some("big");
    let env = |k: &str, d: u8| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    let (cubes, extra) = (env("CUBES", 5), env("EXTRA", 3));
    let dir = std::path::Path::new("tables");
    let mut p1 = load_p1(dir, big, 8, false);
    p1.table.data.collapse();
    if let Some(kind) = &mask {
        let sel = match kind.as_str() {
            "big" => MaskSel::Big,
            "wide" => MaskSel::Succ,
            _ => MaskSel::Reduced,
        };
        load_mask_sel(&mut p1, dir, sel, true);
        p1.mask.as_ref().unwrap().collapse();
    }
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let (mut nodes, mut reads, mut eps) = (0u64, 0u64, 0usize);
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
            eps += found.len();
            found.hash(&mut h);
        }
    }
    println!(
        "endpoints {eps}, hash {:016x}, nodes {nodes}, reads {reads}, {:.3} s",
        h.finish(),
        t0.elapsed().as_secs_f64()
    );
}
