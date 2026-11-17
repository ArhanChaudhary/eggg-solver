//! For phase-1 endpoints of random cubes: phase-2 heuristic components vs optimal phase-2 length.
use eggg_solver::cube::Cube;
use eggg_solver::moves::{MOVE_PERMS, N_MOVES};
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::phase2::{Search2, standardise};
use eggg_solver::solver::load_tables;
use rand::SeedableRng;

fn main() {
    let t = load_tables(std::path::Path::new("tables"), 2, 8, false);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
    let mut gaps = std::collections::BTreeMap::new();
    let mut total_nodes = 0u64;
    let mut l2s = vec![];
    for _ in 0..4 {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = t.p1.distance(&root);
        let mut eps = vec![];
        for len in d0..d0 + 3 {
            let mut s = Search1::new(&t.p1, |p: &[u8], f| {
                eps.push((p.to_vec(), f));
                eps.len() < 40
            });
            s.dfs(&root, d0, len, N_MOVES);
            if eps.len() >= 40 {
                break;
            }
        }
        for (moves, frames) in eps {
            for f in 0..3 {
                if frames >> f & 1 == 0 {
                    continue;
                }
                let mut c = cube;
                for &m in &moves {
                    c = c.apply_perm(&MOVE_PERMS[m as usize]);
                }
                let x = standardise(&c, f);
                let cc = t.p2.centre_coord_of(&x);
                let hc = t.p2.centre_dist[cc as usize];
                let tau = t.p2.tau_of(&x);
                let dt = t.p2.tau_distance(&tau);
                let h = hc.max(dt);
                let mut s2 = Search2::new(&t.p2);
                let mut l2 = None;
                for d in h..20 {
                    let mut found = false;
                    s2.dfs(cc, &tau, dt, d, N_MOVES, &mut |_| {
                        found = true;
                        false
                    });
                    if found {
                        l2 = Some(d);
                        break;
                    }
                }
                let l2 = l2.unwrap();
                total_nodes += s2.nodes;
                l2s.push(l2);
                *gaps.entry((hc, dt, l2)).or_insert(0) += 1;
            }
        }
    }
    for (k, v) in &gaps {
        println!("centres {:2} tau {:2} -> optimal {:2}: {v}", k.0, k.1, k.2);
    }
    println!(
        "mean L2 {:.2}, mean nodes per candidate {:.3e}",
        l2s.iter().map(|&x| x as f64).sum::<f64>() / l2s.len() as f64,
        total_nodes as f64 / l2s.len() as f64
    );
}
