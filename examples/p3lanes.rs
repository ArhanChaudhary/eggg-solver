//! Lanes vs one-search-at-a-time on the same phase-3 tasks: same outcomes, node counts, time.
use eggg_solver::cube3::Cube3;
use eggg_solver::lanes3::{Lanes, Outcome};
use eggg_solver::phase3::{P3Tables, Search3, Task};
use rand::{Rng, SeedableRng};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn main() {
    let k: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(6);
    let nl: usize = std::env::args()
        .nth(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(8);
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), k, 8, false);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(77);
    let mut jobs: Vec<(usize, Task)> = vec![];
    let mut n_searches = 0;
    for _ in 0..12 {
        let mut x = Cube3::SOLVED;
        for _ in 0..60 {
            x = x.apply(rng.gen_range(0..18));
        }
        for d in [14u8, 15, 16] {
            if let Some(ts) = Search3::new(&t).tasks(&x, d) {
                jobs.extend(ts.into_iter().map(|task| (n_searches, task)));
            }
            n_searches += 1;
        }
    }
    let t0 = Instant::now();
    let mut found_a = vec![false; n_searches];
    let mut nodes_a = 0;
    for (s, task) in &jobs {
        let mut srch = Search3::new(&t);
        if srch.run_task(task).is_some() {
            found_a[*s] = true;
        }
        nodes_a += srch.nodes;
    }
    let ta = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let mut found_b = vec![false; n_searches];
    let mut lanes = Lanes::new(&t, nl);
    let stop = AtomicBool::new(false);
    let mut it = jobs.iter().enumerate();
    lanes.run(
        || it.next().map(|(j, (_, task))| (j, task)),
        &stop,
        |o| {
            if let Outcome::Found(j, p) = o {
                found_b[jobs[j].0] = true;
                assert!(!p.is_empty());
            }
        },
    );
    let tb = t1.elapsed().as_secs_f64();
    assert_eq!(found_a, found_b);
    println!(
        "{} tasks: one at a time {nodes_a} nodes {ta:.2} s; {nl} lanes {} nodes {tb:.2} s",
        jobs.len(),
        lanes.nodes
    );
}
