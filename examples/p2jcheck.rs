//! Joint phase-2 table checks: admissibility, dominance over tau, and the phase-2 gap and node
//! counts on real phase-1 endpoints, with and without the joint table.
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES};
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::phase2::{P2_MOVES, Search2, standardise};
use eggg_solver::solver::load_tables_with;
use rand::{Rng, SeedableRng};

fn main() {
    let t = load_tables_with(
        std::path::Path::new("tables"),
        2,
        true,
        false,
        (None, None),
        8,
        false,
    );
    let p2 = &t.p2;
    let j = p2.joint.as_ref().unwrap();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
    // Admissibility on random phase-2 move sequences from solved.
    let mut bad = 0;
    for _ in 0..3000 {
        let k = rng.gen_range(1..14);
        let mut c = Cube::SOLVED;
        for _ in 0..k {
            c = c.apply_perm(&MOVE_PERMS[P2_MOVES[rng.gen_range(0..17)] as usize]);
        }
        let cc = p2.centre_coord_of(&c);
        let tau = p2.tau_of(&c);
        let dj = j.distance(p2, cc / 1680, &tau, cc & 1);
        let dt = p2.tau_distance(&tau);
        if dj as usize > k || dj < dt {
            bad += 1;
        }
    }
    println!("admissibility/dominance violations: {bad}");
    // Gap and nodes on phase-1 endpoints.
    let (mut n_old, mut n_new, mut cnt) = (0u64, 0u64, 0usize);
    let (mut sum_l2, mut sum_ht, mut sum_hj) = (0usize, 0usize, 0usize);
    for _ in 0..4 {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = t.p1.distance(&root);
        let mut eps = vec![];
        for len in d0..d0 + 4 {
            let mut s = Search1::new(&t.p1, |p: &[u8], f| {
                eps.push((p.to_vec(), f));
                eps.len() < 30
            });
            s.dfs(&root, d0, len, N_MOVES);
            if eps.len() >= 30 {
                break;
            }
        }
        for (moves, frames) in eps {
            let f = frames.trailing_zeros() as usize;
            let mut c = cube;
            for &m in &moves {
                c = c.apply_perm(&MOVE_PERMS[m as usize]);
            }
            let x = standardise(&c, f);
            let cc = p2.centre_coord_of(&x);
            let tau = p2.tau_of(&x);
            let dt = p2.tau_distance(&tau);
            let dj = j.distance(p2, cc / 1680, &tau, cc & 1);
            let df = j.distance_fb(p2, cc, &tau);
            let hc = p2.centre_dist[cc as usize];
            let mut l2 = 0;
            let mut s_old = Search2::new(p2);
            for d in hc.max(dt)..25 {
                let mut found = false;
                s_old.dfs(
                    cc,
                    &tau,
                    dt,
                    d,
                    N_MOVES,
                    ALLOWED_AFTER[N_MOVES],
                    &mut |_| {
                        found = true;
                        false
                    },
                );
                if found {
                    l2 = d;
                    break;
                }
            }
            let mut s_new = Search2::new(p2);
            for d in hc.max(dj).max(df)..25 {
                let mut found = false;
                s_new.dfs_joint(
                    cc,
                    &tau,
                    dj,
                    df,
                    d,
                    N_MOVES,
                    ALLOWED_AFTER[N_MOVES],
                    &mut |_| {
                        found = true;
                        false
                    },
                );
                if found {
                    assert_eq!(d, l2, "joint search found a different optimum");
                    break;
                }
            }
            n_old += s_old.nodes;
            n_new += s_new.nodes;
            sum_l2 += l2 as usize;
            sum_ht += hc.max(dt) as usize;
            sum_hj += hc.max(dj).max(df) as usize;
            cnt += 1;
        }
    }
    let c = cnt as f64;
    println!(
        "{cnt} endpoints: optimal L2 {:.2}, old bound {:.2}, joint bound {:.2}",
        sum_l2 as f64 / c,
        sum_ht as f64 / c,
        sum_hj as f64 / c
    );
    println!(
        "nodes to optimal: old {:.3e}, joint {:.3e} per endpoint ({:.1}x fewer)",
        n_old as f64 / c,
        n_new as f64 / c,
        n_old as f64 / n_new as f64
    );
}
