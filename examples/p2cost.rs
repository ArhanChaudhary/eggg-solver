//! Phase-2 cost per level: every phase-2 solution of length hu + slack (hu = max(centre
//! distance, joint U/D distance), the bound before the F/B view) of the candidates of random
//! cubes' phase-1 endpoints, searched with `dfs_joint` as the solver does. Prints, per slack,
//! the solutions, nodes and thread CPU time per solution, and a hash of all solutions in order
//! (equal hashes: the same solutions in the same order). `threads` threads take the
//! candidates one at a time (CPU times are per thread; the wall time shows the throughput).
//! Usage: p2cost [cubes] [endpoints per cube] [max slack] [seed] [threads] (tables in
//! $EGGG_TABLES or ./tables).
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES};
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::phase2::{Search2, standardise};
use eggg_solver::solver::{load_p1, load_p2};
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
    let arg = |i: usize, d: u64| -> u64 {
        std::env::args()
            .nth(i)
            .map(|s| s.parse().unwrap())
            .unwrap_or(d)
    };
    let (ncubes, per_cube, max_slack, seed, threads) = (
        arg(1, 8) as usize,
        arg(2, 60) as usize,
        arg(3, 3) as u8,
        arg(4, 7),
        arg(5, 1).max(1) as usize,
    );
    let dir = std::env::var_os("EGGG_TABLES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "tables".into());
    let p1 = load_p1(&dir, false, 4, false);
    let p2 = &load_p2(&dir, true, 4, false);
    let j = p2.joint.as_ref().unwrap();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    // (standardised cube, centre coordinate, tau) of each candidate.
    let mut cands = vec![];
    for _ in 0..ncubes {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        let mut eps = vec![];
        for len in d0..d0 + 6 {
            let mut s = Search1::new(&p1, |p: &[u8], f| {
                eps.push((p.to_vec(), f));
                eps.len() < per_cube
            });
            s.dfs(&root, d0, len, N_MOVES);
            if eps.len() >= per_cube {
                break;
            }
        }
        for (moves, frames) in eps {
            let mut c = cube;
            for &m in &moves {
                c = c.apply_perm(&MOVE_PERMS[m as usize]);
            }
            for f in 0..3 {
                if frames >> f & 1 == 1 {
                    let x = standardise(&c, f);
                    let cc = p2.centre_coord_of(&x);
                    if p2.centre_dist[cc as usize] != eggg_solver::table::UNSEEN {
                        cands.push((cc, p2.tau_of(&x)));
                    }
                }
            }
        }
    }
    let n = max_slack as usize + 1;
    // Per candidate: (U/D bound, bound, per slack (searched, solutions, nodes, CPU s), the
    // solutions' moves in order, 0xff after each).
    type Res = (u8, u8, Vec<(u64, u64, u64, f64)>, Vec<u8>);
    let run = |&(cc, tau): &(u32, [u8; 16])| -> Res {
        let hc = p2.centre_dist[cc as usize];
        let dj = j.distance(cc / 1680, &tau, cc & 1);
        let df = j.distance_fb(cc, &tau); // fb-only
        let hu = hc.max(dj);
        let h2 = hu;
        let h2 = h2.max(df); // fb-only
        let mut per = vec![(0, 0, 0, 0.0); n];
        let mut moves = vec![];
        for s in 0..=max_slack {
            let l2 = hu + s;
            if l2 < h2 {
                continue;
            }
            let t0 = cpu_now();
            let mut srch = Search2::new(p2);
            let mut found = 0u64;
            srch.dfs_joint(
                cc,
                &tau,
                dj,
                df, // fb-only
                l2,
                N_MOVES,
                ALLOWED_AFTER[N_MOVES],
                &mut |p| {
                    found += 1;
                    moves.extend_from_slice(p);
                    moves.push(0xff);
                    true
                },
            );
            per[s as usize] = (1, found, srch.nodes, cpu_now() - t0);
        }
        (hu, h2, per, moves)
    };
    // Candidates taken by the threads one at a time.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let t_wall = std::time::Instant::now();
    let mut res: Vec<Option<Res>> = (0..cands.len()).map(|_| None).collect();
    let (total, wall) = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                let (cands, run, next) = (&cands, &run, &next);
                sc.spawn(move || {
                    let t0 = cpu_now();
                    let mut out: Vec<(usize, Res)> = vec![];
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= cands.len() {
                            break;
                        }
                        out.push((i, run(&cands[i])));
                    }
                    (out, cpu_now() - t0)
                })
            })
            .collect();
        let mut total = 0.0;
        for h in hs {
            let (out, t) = h.join().unwrap();
            total += t;
            for (i, r) in out {
                res[i] = Some(r);
            }
        }
        (total, t_wall.elapsed().as_secs_f64())
    });
    let (mut sols, mut nodes, mut cpu, mut searched) =
        (vec![0u64; n], vec![0u64; n], vec![0f64; n], vec![0u64; n]);
    let (mut sum_hu, mut sum_h2) = (0usize, 0usize);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (hu, h2, per, moves) in res.into_iter().map(Option::unwrap) {
        sum_hu += hu as usize;
        sum_h2 += h2 as usize;
        for (s, &(a, b, c, d)) in per.iter().enumerate() {
            searched[s] += a;
            sols[s] += b;
            nodes[s] += c;
            cpu[s] += d;
        }
        for m in moves {
            hash = (hash ^ m as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    let c = cands.len() as f64;
    println!(
        "{} candidates: mean bound hu {:.3}, h2 {:.3}; {threads} threads: total CPU {total:.2} s, \
         wall {wall:.2} s; solution hash {hash:016x}",
        cands.len(),
        sum_hu as f64 / c,
        sum_h2 as f64 / c
    );
    println!("| slack | searches | solutions | nodes | nodes/solution | CPU s | us/solution |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for s in 0..n {
        println!(
            "| {s} | {} | {} | {} | {:.0} | {:.3} | {:.2} |",
            searched[s],
            sols[s],
            nodes[s],
            nodes[s] as f64 / sols[s].max(1) as f64,
            cpu[s],
            cpu[s] * 1e6 / sols[s].max(1) as f64
        );
    }
}
