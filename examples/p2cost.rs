//! Cost per reduction at deep phase-2 levels, split into the phase-2 search (joint table,
//! both views), the solver's per-solution work (apply, reduce to a 3x3, H48 bound)
//! and phase 3 (all depths from the bound up to `margin - 1`, as the driver does for a
//! reduction at L12 = best - margin). Thread CPU time (the machine may be shared).
//! Usage: p2cost [positions] [slack] [h48 K] [margin]
use eggg_solver::cube::Cube;
use eggg_solver::cube3::Cube3;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES};
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::phase2::{P2_MOVES, Search2, standardise};
use eggg_solver::phase3::Search3;
use eggg_solver::solver::load_tables_with;
use rand::SeedableRng;

fn cpu_now() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let npos: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(4);
    let slack: u8 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2);
    let k: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(7);
    let margin: u8 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(13);
    let t = load_tables_with(
        std::path::Path::new("tables"),
        k,
        true,
        false,
        (None, None),
        4,
        false,
    );
    let p2 = &t.p2;
    let j = p2.joint.as_ref().unwrap();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
    let mut cands = vec![];
    for _ in 0..npos {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = t.p1.distance(&root);
        let mut eps = vec![];
        for len in d0..d0 + 5 {
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
            let mut c = cube;
            for &m in &moves {
                c = c.apply_perm(&MOVE_PERMS[m as usize]);
            }
            let x = standardise(&c, frames.trailing_zeros() as usize);
            let cc = p2.centre_coord_of(&x);
            if p2.centre_dist[cc as usize] != eggg_solver::table::UNSEEN {
                cands.push((x, cc, p2.tau_of(&x)));
            }
        }
    }
    let (mut t_both, mut t_post, mut t_p3) = (0f64, 0f64, 0f64);
    let (mut sols, mut kept, mut n_both, mut n3) = (0u64, 0u64, 0u64, 0u64);
    let _ = P2_MOVES;
    for (x, cc, tau) in &cands {
        let (cc, tau) = (*cc, *tau);
        let hc = p2.centre_dist[cc as usize];
        let dj = j.distance(p2, cc / 1680, &tau, cc & 1);
        let df = j.distance_fb(p2, cc, &tau);
        let l2 = hc.max(dj).max(df) + slack;
        let t0 = cpu_now();
        let mut s = Search2::new(p2);
        let mut paths = vec![];
        s.dfs_joint(
            cc,
            &tau,
            dj,
            df,
            l2,
            N_MOVES,
            ALLOWED_AFTER[N_MOVES],
            &mut |p| {
                paths.push(p.to_vec());
                true
            },
        );
        t_both += cpu_now() - t0;
        n_both += s.nodes;
        sols += paths.len() as u64;
        // The solver's per-solution work, then phase 3 to depth margin - 1.
        let t0 = cpu_now();
        let mut reds = vec![];
        for p in &paths {
            let mut y = *x;
            for &m in p {
                y = y.apply_perm(&MOVE_PERMS[m as usize]);
            }
            if let Some(c3) = Cube3::from_reduced(&y) {
                let h3 = Search3::new(&t.p3).lower_bound(&c3);
                if h3 < margin {
                    reds.push((c3, h3));
                }
            }
        }
        t_post += cpu_now() - t0;
        kept += reds.len() as u64;
        let t0 = cpu_now();
        for (c3, h3) in &reds {
            let mut s3 = Search3::new(&t.p3);
            for d in *h3..margin {
                if s3.search_depth(c3, d).is_some() {
                    break;
                }
            }
            n3 += s3.nodes;
        }
        t_p3 += cpu_now() - t0;
    }
    let per = |x: f64| x / sols.max(1) as f64 * 1e6;
    println!(
        "{} endpoints, slack {slack}: {sols} phase-2 solutions, {kept} with h3 < {margin} (h48 h{k})",
        cands.len()
    );
    println!(
        "CPU per solution: phase 2 (joint table, both views) {:.1} us ({:.0} nodes), \
         per-solution work {:.1} us, phase 3 to depth {} {:.1} us ({:.0} nodes)",
        per(t_both),
        n_both as f64 / sols.max(1) as f64,
        per(t_post),
        margin - 1,
        per(t_p3),
        n3 as f64 / sols.max(1) as f64
    );
}
