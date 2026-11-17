//! Phase-1 lanes vs Search1 on the same jobs: same endpoints, and the time of each.
use eggg_solver::cube::Cube;
use eggg_solver::lanes1::{Job, Lanes1};
use eggg_solver::moves::{ALLOWED_AFTER, N_MOVES};
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::solver::load_tables;
use eggg_solver::table::Mod3Table;
use rand::SeedableRng;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

fn main() {
    let nl: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(4);
    let t = load_tables(std::path::Path::new("tables"), 2, 8, false);
    let p1 = &t.p1;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
    let (mut ta, mut tb) = (0.0, 0.0);
    let (mut na, mut nb) = (0u64, 0u64);
    for _ in 0..5 {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        for len in d0..=d0 + 3 {
            // jobs: 2-move prefixes that can still reach the goal
            let mut jobs: Vec<(Vec<u8>, P1State, u8)> = vec![];
            for a in 0..N_MOVES {
                for b in 0..N_MOVES {
                    if ALLOWED_AFTER[a] >> b & 1 == 0 {
                        continue;
                    }
                    let mut st = root;
                    let mut d = d0;
                    let mut ok = true;
                    for (k, &m) in [a, b].iter().enumerate() {
                        st = st.apply(m, p1);
                        d = Mod3Table::decode(d, p1.value(&st));
                        if d as usize > len as usize - k - 1 {
                            ok = false;
                            break;
                        }
                    }
                    if ok && len >= 2 {
                        jobs.push((vec![a as u8, b as u8], st, d));
                    }
                }
            }
            let t0 = Instant::now();
            let mut ea = vec![];
            for (p, st, d) in &jobs {
                let mut s = Search1::new(p1, |q: &[u8], f| {
                    ea.push((q.to_vec(), f));
                    true
                });
                s.path.extend_from_slice(p);
                s.dfs(st, *d, len - 2, p[1] as usize);
                na += s.nodes;
            }
            ta += t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let mut eb = vec![];
            let mut ln = Lanes1::new(p1, nl, root);
            let stop = AtomicBool::new(false);
            let mut it = jobs.iter().enumerate();
            ln.run(
                || {
                    it.next().map(|(j, (p, st, d))| {
                        (
                            j,
                            Job {
                                state: *st,
                                d: *d,
                                left: len - 2,
                                prefix: p,
                            },
                        )
                    })
                },
                &stop,
                |_, q, f| eb.push((q.to_vec(), f)),
                |_| {},
            );
            nb += ln.nodes;
            tb += t1.elapsed().as_secs_f64();
            ea.sort();
            eb.sort();
            assert_eq!(ea, eb, "endpoints differ at length {len}");
        }
    }
    println!("Search1 {na} nodes {ta:.2} s; {nl} lanes {nb} nodes {tb:.2} s");
}
