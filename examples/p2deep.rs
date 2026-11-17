//! Deep phase-2 enumeration: nodes per solution at L2 = h2 + slack, with the current bound
//! (exact centres, joint table on the U/D pattern) and with the joint table also read for the
//! F/B pattern. Also counts "useful" nodes (whose subtree holds a solution): what a perfect
//! bound would expand. Usage: p2deep [positions] [max slack] [threads]
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES, PHASE2_MASK};
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::phase2::{P2_MOVES, P2Tables, WIDE_HALF, standardise};
use eggg_solver::solver::load_tables_with;
use eggg_solver::table::Mod3Table;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Default, Clone, Copy)]
struct St {
    nodes: u64,
    useful: u64,
    sols: u64,
    lookups: u64,
}

fn last_mask(last: usize) -> u32 {
    if last < 18 {
        WIDE_HALF & !(1 << (19 + 3 * ((last / 3) % 3)))
    } else {
        WIDE_HALF
    }
}

const AXIS_MOVES: [u32; 3] = [0x1c0e07, 0xe07038, 0x70381c0];
fn after_child(after: u32, m: usize) -> u32 {
    ALLOWED_AFTER[m] & (after | !AXIS_MOVES[(m / 3) % 3])
}

/// Is a state with (U/D, F/B) joint distances (dj, df) within `left` moves of a state with
/// U/D and F/B centres solved, tau solved and no PLL parity (the R/L centres ignored)?
#[allow(clippy::too_many_arguments)]
fn proj_search(
    t: &P2Tables,
    cc: u32,
    tau: &[u8; 16],
    dj: u8,
    df: u8,
    left: u8,
    last: usize,
) -> bool {
    if dj == 0 && df == 0 {
        return true;
    }
    if left == 0 {
        return false;
    }
    let j = t.joint.as_ref().unwrap();
    let allowed = ALLOWED_AFTER[last] & PHASE2_MASK;
    for (i, &m) in P2_MOVES.iter().enumerate() {
        if allowed >> m & 1 == 0 {
            continue;
        }
        let c2 = t.centre_move[cc as usize][i];
        let t2 = t.tau_move(tau, i);
        let d2 = Mod3Table::decode(dj, j.get(j.index(c2 / 1680, &t2, c2 & 1)));
        if d2 >= left {
            continue;
        }
        let f2 = Mod3Table::decode(df, j.get(j.index_fb(c2, &t2)));
        if f2 >= left {
            continue;
        }
        if proj_search(t, c2, &t2, d2, f2, left - 1, m as usize) {
            return true;
        }
    }
    false
}

type Memo = std::collections::HashMap<(u32, [u8; 16]), u8>;

/// Exact distance in the (U/D, F/B, tau, PLL) projection: what a joint table over all four
/// would give.
fn proj_dist(t: &P2Tables, cc: u32, tau: &[u8; 16], dj: u8, df: u8, memo: &mut Memo) -> u8 {
    // The R/L pattern does not matter: key on cc with rl = 0.
    let key = (cc / 24 * 24 + (cc & 1), *tau);
    if let Some(&d) = memo.get(&key) {
        return d;
    }
    let mut b = dj.max(df);
    while !proj_search(t, cc, tau, dj, df, b, N_MOVES) {
        b += 1;
    }
    memo.insert(key, b);
    b
}

/// [`dfs`] pruned by the exact centre distance and the exact projected distance.
#[allow(clippy::too_many_arguments)]
fn dfs_oracle(
    t: &P2Tables,
    cc: u32,
    tau: &[u8; 16],
    dj: u8,
    df: u8,
    left: u8,
    last: usize,
    after: u32,
    st: &mut St,
    memo: &mut Memo,
) -> u64 {
    st.nodes += 1;
    if left == 0 {
        let s = (t.centre_dist[cc as usize] == 0 && dj == 0) as u64;
        st.sols += s;
        st.useful += s;
        return s;
    }
    let j = t.joint.as_ref().unwrap();
    let allowed = after & PHASE2_MASK & if left == 1 { last_mask(last) } else { u32::MAX };
    let mut total = 0;
    for (i, &m) in P2_MOVES.iter().enumerate() {
        if allowed >> m & 1 == 0 {
            continue;
        }
        let c2 = t.centre_move[cc as usize][i];
        if t.centre_dist[c2 as usize] >= left {
            continue;
        }
        let t2 = t.tau_move(tau, i);
        let d2 = Mod3Table::decode(dj, j.get(j.index(c2 / 1680, &t2, c2 & 1)));
        let f2 = Mod3Table::decode(df, j.get(j.index_fb(c2, &t2)));
        if d2 >= left || f2 >= left {
            continue;
        }
        st.lookups += 1;
        if proj_dist(t, c2, &t2, d2, f2, memo) >= left {
            continue;
        }
        total += dfs_oracle(
            t,
            c2,
            &t2,
            d2,
            f2,
            left - 1,
            m as usize,
            after_child(after, m as usize),
            st,
            memo,
        );
    }
    if total > 0 {
        st.useful += 1;
    }
    total
}

/// Returns the number of solutions below.
#[allow(clippy::too_many_arguments)]
fn dfs(
    t: &P2Tables,
    fb: bool,
    cc: u32,
    tau: &[u8; 16],
    dj: u8,
    df: u8,
    left: u8,
    last: usize,
    after: u32,
    st: &mut St,
) -> u64 {
    st.nodes += 1;
    if left == 0 {
        let s = (t.centre_dist[cc as usize] == 0 && dj == 0) as u64;
        st.sols += s;
        st.useful += s;
        return s;
    }
    let j = t.joint.as_ref().unwrap();
    let allowed = after & PHASE2_MASK & if left == 1 { last_mask(last) } else { u32::MAX };
    let mut kids = [(0u32, [0u8; 16], 0usize, 0usize, 0u8); 17];
    let mut nk = 0;
    for (i, &m) in P2_MOVES.iter().enumerate() {
        if allowed >> m & 1 == 0 {
            continue;
        }
        let c2 = t.centre_move[cc as usize][i];
        if t.centre_dist[c2 as usize] >= left {
            continue;
        }
        let t2 = t.tau_move(tau, i);
        let idx = j.index(c2 / 1680, &t2, c2 & 1);
        j.prefetch(idx);
        let idf = if fb {
            let x = j.index_fb(c2, &t2);
            j.prefetch(x);
            x
        } else {
            0
        };
        kids[nk] = (c2, t2, idx, idf, m);
        nk += 1;
    }
    let mut total = 0;
    for &(c2, t2, idx, idf, m) in &kids[..nk] {
        st.lookups += 1;
        let d2 = Mod3Table::decode(dj, j.get(idx));
        if d2 >= left {
            continue;
        }
        let f2 = if fb {
            st.lookups += 1;
            let f2 = Mod3Table::decode(df, j.get(idf));
            if f2 >= left {
                continue;
            }
            f2
        } else {
            0
        };
        total += dfs(
            t,
            fb,
            c2,
            &t2,
            d2,
            f2,
            left - 1,
            m as usize,
            after_child(after, m as usize),
            st,
        );
    }
    if total > 0 {
        st.useful += 1;
    }
    total
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let npos: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(6);
    let max_slack: u8 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let threads: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(6);
    let t = load_tables_with(
        std::path::Path::new("tables"),
        0,
        true,
        false,
        threads,
        false,
    );
    let p2 = &t.p2;
    let j = p2.joint.as_ref().unwrap();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
    // Admissibility of the F/B view on random phase-2 sequences.
    let mut bad = 0;
    for _ in 0..20000 {
        let k = rng.gen_range(1..16);
        let mut c = Cube::SOLVED;
        for _ in 0..k {
            c = c.apply_perm(&MOVE_PERMS[P2_MOVES[rng.gen_range(0..17)] as usize]);
        }
        let cc = p2.centre_coord_of(&c);
        let tau = p2.tau_of(&c);
        let df = j.distance_fb(p2, cc, &tau);
        if df as usize > k {
            bad += 1;
        }
    }
    println!("F/B view admissibility violations: {bad} of 20000");
    // Candidates: phase-1 endpoints in their frames.
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
            let f = frames.trailing_zeros() as usize;
            let mut c = cube;
            for &m in &moves {
                c = c.apply_perm(&MOVE_PERMS[m as usize]);
            }
            let x = standardise(&c, f);
            let cc = p2.centre_coord_of(&x);
            if p2.centre_dist[cc as usize] == eggg_solver::table::UNSEEN {
                continue;
            }
            cands.push((cc, p2.tau_of(&x)));
        }
    }
    println!("{} candidates", cands.len());
    // Bounds.
    let (mut s_old, mut s_new, mut n_gt) = (0usize, 0usize, 0usize);
    let mut info = vec![];
    for &(cc, tau) in &cands {
        let hc = p2.centre_dist[cc as usize];
        let dj = j.distance(p2, cc / 1680, &tau, cc & 1);
        let df = j.distance_fb(p2, cc, &tau);
        let ho = hc.max(dj);
        let hn = ho.max(df);
        s_old += ho as usize;
        s_new += hn as usize;
        n_gt += (hn > ho) as usize;
        info.push((cc, tau, dj, df, ho));
    }
    let n = cands.len() as f64;
    println!(
        "bound: old {:.3}, with F/B {:.3}; tighter on {:.1}% of candidates",
        s_old as f64 / n,
        s_new as f64 / n,
        100.0 * n_gt as f64 / n
    );
    let oracle = std::env::var_os("ORACLE").is_some();
    if oracle {
        // Optimal L2 and the projected (U/D x F/B x tau x PLL) distance.
        let mut memo = Memo::new();
        let (mut s_proj, mut s_big, mut s_opt) = (0usize, 0usize, 0usize);
        let mut gaps = [0usize; 8];
        for &(cc, tau, dj, df, _) in &info {
            let hc = p2.centre_dist[cc as usize];
            let dp = proj_dist(p2, cc, &tau, dj, df, &mut memo);
            let hb = hc.max(dp);
            let mut l2 = hb;
            loop {
                let mut st = St::default();
                if dfs(
                    p2,
                    true,
                    cc,
                    &tau,
                    dj,
                    df,
                    l2,
                    N_MOVES,
                    ALLOWED_AFTER[N_MOVES],
                    &mut st,
                ) > 0
                {
                    break;
                }
                l2 += 1;
            }
            s_proj += dp as usize;
            s_big += hb as usize;
            s_opt += l2 as usize;
            gaps[((l2 - hb) as usize).min(7)] += 1;
        }
        println!(
            "projected distance {:.3}, max(centres, projected) {:.3}, optimal L2 {:.3}; gaps optimal - bound: {:?}",
            s_proj as f64 / n,
            s_big as f64 / n,
            s_opt as f64 / n,
            gaps
        );
    }
    for slack in 0..=max_slack {
        let res = Mutex::new([St::default(); 3]);
        let times = Mutex::new([0f64; 3]);
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut loc = [St::default(); 3];
                    let mut tm = [0f64; 3];
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if k >= info.len() {
                            break;
                        }
                        let (cc, tau, dj, df, ho) = info[k];
                        let l2 = ho + slack;
                        let mut sols = [0; 2];
                        for (v, fb) in [false, true].into_iter().enumerate() {
                            let t0 = Instant::now();
                            sols[v] = dfs(
                                p2,
                                fb,
                                cc,
                                &tau,
                                dj,
                                df,
                                l2,
                                N_MOVES,
                                ALLOWED_AFTER[N_MOVES],
                                &mut loc[v],
                            );
                            tm[v] += t0.elapsed().as_secs_f64();
                        }
                        assert_eq!(sols[0], sols[1]);
                        if oracle {
                            let t0 = Instant::now();
                            let mut memo = Memo::new();
                            let so = dfs_oracle(
                                p2,
                                cc,
                                &tau,
                                dj,
                                df,
                                l2,
                                N_MOVES,
                                ALLOWED_AFTER[N_MOVES],
                                &mut loc[2],
                                &mut memo,
                            );
                            tm[2] += t0.elapsed().as_secs_f64();
                            assert_eq!(so, sols[0]);
                        }
                    }
                    let mut r = res.lock().unwrap();
                    let mut ti = times.lock().unwrap();
                    for v in 0..3 {
                        r[v].nodes += loc[v].nodes;
                        r[v].useful += loc[v].useful;
                        r[v].sols += loc[v].sols;
                        r[v].lookups += loc[v].lookups;
                        ti[v] += tm[v];
                    }
                });
            }
        });
        let r = res.into_inner().unwrap();
        let ti = times.into_inner().unwrap();
        for (v, name) in ["UD", "UD+FB", "oracle"].iter().enumerate() {
            if v == 2 && !oracle {
                continue;
            }
            let x = r[v];
            println!(
                "slack {slack} {name:6}: sols {:9}, nodes {:11} ({:7.1}/sol), useful {:9} ({:4.1}%), lookups {:11}, {:.2} s ({:.2} us/sol)",
                x.sols,
                x.nodes,
                x.nodes as f64 / x.sols.max(1) as f64,
                x.useful,
                100.0 * x.useful as f64 / x.nodes as f64,
                x.lookups,
                ti[v],
                1e6 * ti[v] / x.sols.max(1) as f64
            );
        }
    }
}
