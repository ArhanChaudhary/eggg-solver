//! Checks for the large joint phase-2 tables once they are generated: their distances against
//! the exact projected distances computed by IDA* (examples/common/proj.rs) on sampled
//! states, and the phase-2 search with them against the joint-table search (same solutions,
//! fewer nodes).
//! Usage: p2bigcheck KINDS [positions] [max slack]   (KINDS: udrl, udfb or udrl,udfb)
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES};
use eggg_solver::p2big::Kind;
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::phase2::{MAX_BIG_SRC, P2_MOVES, Search2, standardise};
use eggg_solver::solver::load_tables_full;
use rand::{Rng, SeedableRng};

#[path = "common/proj.rs"]
mod proj;
use proj::{Memo, Proj};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kinds: Vec<Kind> = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("udrl")
        .split(',')
        .map(|k| match k {
            "udrl" => Kind::UdRl,
            "udfb" => Kind::UdFb,
            _ => panic!("unknown kind {k}"),
        })
        .collect();
    let npos: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4);
    let max_slack: u8 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2);
    let t = load_tables_full(
        std::path::Path::new("tables"),
        0,
        true,
        false,
        &kinds,
        6,
        true,
    );
    let p2 = &t.p2;
    let j = p2.joint.as_ref().unwrap();
    let pj = Proj::new(p2);
    let mut memo = Memo::new();
    // Projection of each bound source: U/D x R/L direct 1, its view 2, U/D x F/B 0.
    let proj_of: Vec<u8> = p2
        .big_src
        .iter()
        .map(|&(i, view)| match (p2.bigs[i].kind, view) {
            (Kind::UdRl, false) => 1,
            (Kind::UdRl, true) => 2,
            (Kind::UdFb, _) => 0,
        })
        .collect();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(11);
    // 1. Random phase-2 states.
    let mut bad = 0;
    let mut checked = 0;
    for _ in 0..3000 {
        let k = rng.gen_range(1..20);
        let mut c = Cube::SOLVED;
        for _ in 0..k {
            c = c.apply_perm(&MOVE_PERMS[P2_MOVES[rng.gen_range(0..17)] as usize]);
        }
        let cc = p2.centre_coord_of(&c);
        let tau = p2.tau_of(&c);
        let dj = j.distance(p2, cc / 1680, &tau, cc & 1);
        let df = j.distance_fb(p2, cc, &tau);
        for (s, &pk) in proj_of.iter().enumerate() {
            let d = p2.big_distance(s, cc, &tau);
            let want = pj.dist(pk, cc, &tau, dj, df, &mut memo);
            checked += 1;
            if d != want || d as usize > k {
                bad += 1;
            }
        }
    }
    println!("random phase-2 states: {bad} of {checked} table distances differ from IDA*");
    // 2. Phase-1 endpoints: distances, and the search against the joint table's.
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
                cands.push((cc, p2.tau_of(&x)));
            }
        }
    }
    let mut bad = 0;
    let (mut h_joint, mut h_big) = (0usize, 0usize);
    let mut infos = vec![];
    for &(cc, tau) in &cands {
        let hc = p2.centre_dist[cc as usize];
        let dj = j.distance(p2, cc / 1680, &tau, cc & 1);
        let df = j.distance_fb(p2, cc, &tau);
        let mut ds = [0u8; MAX_BIG_SRC];
        for (s, &pk) in proj_of.iter().enumerate() {
            ds[s] = p2.big_distance(s, cc, &tau);
            if ds[s] != pj.dist(pk, cc, &tau, dj, df, &mut memo) {
                bad += 1;
            }
        }
        let hj = hc.max(dj).max(df);
        h_joint += hj as usize;
        h_big += ds.iter().fold(hj, |a, &b| a.max(b)) as usize;
        infos.push((cc, tau, dj, df, ds, hj));
    }
    let n = cands.len() as f64;
    println!(
        "{} endpoints: {bad} distances differ from IDA*; bound {:.3} (joint) -> {:.3}",
        cands.len(),
        h_joint as f64 / n,
        h_big as f64 / n
    );
    for slack in 0..=max_slack {
        let (mut sj, mut sb, mut nj, mut nb) = (0u64, 0u64, 0u64, 0u64);
        for &(cc, tau, dj, df, ds, hj) in &infos {
            let l2 = hj + slack;
            let mut s1 = Search2::new(p2);
            let mut cnt = 0u64;
            s1.dfs_joint(
                cc,
                &tau,
                dj,
                df,
                l2,
                N_MOVES,
                ALLOWED_AFTER[N_MOVES],
                &mut |_| {
                    cnt += 1;
                    true
                },
            );
            sj += cnt;
            nj += s1.nodes;
            let mut s2 = Search2::new(p2);
            let mut cnt2 = 0u64;
            s2.dfs_big(
                cc,
                &tau,
                ds,
                l2,
                N_MOVES,
                ALLOWED_AFTER[N_MOVES],
                &mut |_| {
                    cnt2 += 1;
                    true
                },
            );
            assert_eq!(cnt, cnt2, "different solution counts");
            sb += cnt2;
            nb += s2.nodes;
        }
        println!(
            "slack {slack}: {sj} solutions; nodes per solution joint {:.1}, large tables {:.1} ({:.1}x fewer)",
            nj as f64 / sj.max(1) as f64,
            nb as f64 / sb.max(1) as f64,
            nj as f64 / nb.max(1) as f64
        );
    }
}
