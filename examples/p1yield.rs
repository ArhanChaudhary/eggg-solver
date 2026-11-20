//! Phase-1 job yields at a fine granularity, for job-ordering studies (opt/p1-order2).
//!
//! For the bench positions of seed SEED (POS: comma-separated 0-based indices, default the
//! first CUBES) and the phase-1 lengths LENS, every job the driver makes at that length (a
//! 3-move prefix over all pairs, kept while its staging distance allows the length) whose
//! prefix is sampled (1 in FRAC, by a hash of the prefix: the same prefixes at every length
//! and cube) is searched as the driver's `p1_run` does (its two levels below the prefix
//! expanded here), each 5-move sub-prefix by its own `Search1`. Each endpoint's candidates
//! (frames with L1 + h2 <= LIMIT + 2) get their exact phase-2 bound h2.
//!
//! Output (OUT, binary, 32-byte little-endian records), one per (cube, length, prefix):
//! u8 cube, u8 len, u8 prefix length (3: the job's own expansion nodes; 5: a sub-prefix),
//! u8 staging distance at the prefix, u32 prefix moves (5 bits each, first move lowest, 31 =
//! none), u32 nodes, u32 endpoints, u16 x 6 candidates with L1 + h2 = LIMIT - 3 (or less),
//! LIMIT - 2, .., LIMIT + 2, u8 paired dedges at the prefix, u8 fewest bad dedges over the
//! frames (both wings low or both high), u16 0.
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, N_MOVES};
use eggg_solver::phase1::{ALL_PAIRS, P1State, Search1};
use eggg_solver::solver::{candidate_bounds, load_mask, load_p1, load_p2};
use eggg_solver::table::Mod3Table;
use rand::SeedableRng;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// (paired dedges, fewest bad dedges over the frames) of a state.
fn features(s: &P1State, low: &[[bool; 24]; 3]) -> (u8, u8) {
    let pos = &s.pos.0;
    let mut paired = 0;
    let mut bad = [0u8; 3];
    for e in 0..12 {
        let (a, b) = (pos[2 * e] as usize, pos[2 * e + 1] as usize);
        if a / 2 == b / 2 {
            paired += 1;
        }
        for f in 0..3 {
            if low[f][a] == low[f][b] {
                bad[f] += 1;
            }
        }
    }
    (paired, *bad.iter().min().unwrap())
}

#[allow(clippy::too_many_arguments)]
fn record(
    out: &mut Vec<u8>,
    cube: usize,
    len: u8,
    prefix: &[u8],
    d: u8,
    nodes: u64,
    eps: usize,
    c: &[u16; 6],
    f: (u8, u8),
) {
    let mut code = 0u32;
    for i in 0..5 {
        code |= (*prefix.get(i).unwrap_or(&31) as u32) << (5 * i);
    }
    out.extend_from_slice(&[cube as u8, len, prefix.len() as u8, d]);
    out.extend_from_slice(&code.to_le_bytes());
    out.extend_from_slice(&(nodes.min(u32::MAX as u64) as u32).to_le_bytes());
    out.extend_from_slice(&(eps.min(u32::MAX as usize) as u32).to_le_bytes());
    for x in c {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out.extend_from_slice(&[f.0, f.1, 0, 0]);
}

fn main() {
    let seed: u64 = env("SEED", 6060);
    let cubes: usize = env("CUBES", 4);
    let pos: Vec<usize> = std::env::var("POS").map_or((0..cubes).collect(), |v| {
        v.split(',').map(|x| x.parse().unwrap()).collect()
    });
    let lens: Vec<u8> = std::env::var("LENS")
        .unwrap_or("14,15".into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let frac: u64 = env("FRAC", 8);
    let threads: usize = env("THREADS", 8);
    let limit: usize = env("LIMIT", 26);
    let out_path = std::env::var("OUT").unwrap_or("p1yield.bin".into());
    let dir = std::path::PathBuf::from(std::env::var("TABLES").unwrap_or("tables".into()));
    let mut p1 = load_p1(&dir, false, threads, true);
    p1.table.data.collapse();
    load_mask(&mut p1, &dir, false, true);
    p1.mask.as_ref().unwrap().collapse();
    let p2 = load_p2(&dir, true, threads, true);
    p2.joint.as_ref().unwrap().table.collapse();
    let low: [[bool; 24]; 3] = std::array::from_fn(|f| {
        let mut l = [false; 24];
        for &s in &p1.frames[f].low_slots {
            l[s as usize] = true;
        }
        l
    });
    // The bench positions (as `bench --seed`).
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    let all: Vec<Cube> = (0..=*pos.iter().max().unwrap())
        .map(|_| Cube::random(&mut rng))
        .collect();
    let mut file = std::fs::File::create(&out_path).unwrap();
    for &ci in &pos {
        let cube = all[ci];
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        for &len in &lens {
            let t0 = std::time::Instant::now();
            // The driver's jobs at this length (`Run::level_jobs`), sampled.
            let mut items = vec![];
            for m0 in 0..N_MOVES {
                for m1 in 0..N_MOVES {
                    if ALLOWED_AFTER[m0] >> m1 & 1 == 0 {
                        continue;
                    }
                    for m2 in 0..N_MOVES {
                        if ALLOWED_AFTER[m1] >> m2 & 1 == 0 {
                            continue;
                        }
                        let pid = (m0 * 729 + m1 * 27 + m2) as u64;
                        let h = pid.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40;
                        if !h.is_multiple_of(frac) {
                            continue;
                        }
                        let p = [m0 as u8, m1 as u8, m2 as u8];
                        let mut st = root;
                        let mut d = d0;
                        let mut ok = true;
                        for (i, &m) in p.iter().enumerate() {
                            st = st.apply(m as usize, &p1);
                            d = Mod3Table::decode(d, p1.value(&st));
                            if d as usize > len as usize - i - 1 {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            items.push((p, st, d));
                        }
                    }
                }
            }
            let next = AtomicUsize::new(0);
            let out = Mutex::new(vec![]);
            let totals = Mutex::new((0u64, 0usize, [0u64; 6]));
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut buf = vec![];
                        let (mut tn, mut te, mut tc) = (0u64, 0usize, [0u64; 6]);
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= items.len() {
                                break;
                            }
                            let (p, st, d) = items[i];
                            // The job's two levels below the prefix, as `p1_run` expands them.
                            let mut subs = vec![];
                            let mut own = 0u64;
                            let mut a = ALLOWED_AFTER[p[2] as usize];
                            while a != 0 {
                                let m3 = a.trailing_zeros() as usize;
                                a &= a - 1;
                                let c3 = st.apply(m3, &p1);
                                let d3 = Mod3Table::decode(d, p1.value(&c3));
                                own += 1;
                                if d3 >= len - 3 {
                                    continue;
                                }
                                let mut b = ALLOWED_AFTER[m3];
                                while b != 0 {
                                    let m4 = b.trailing_zeros() as usize;
                                    b &= b - 1;
                                    let c4 = c3.apply(m4, &p1);
                                    let d4 = Mod3Table::decode(d3, p1.value(&c4));
                                    own += 1;
                                    if d4 >= len - 4 {
                                        continue;
                                    }
                                    subs.push(([p[0], p[1], p[2], m3 as u8, m4 as u8], c4, d4));
                                }
                            }
                            let none = [0u16; 6];
                            record(&mut buf, ci, len, &p, d, own, 0, &none, features(&st, &low));
                            tn += own;
                            for (q, c, dq) in subs {
                                let mut eps = vec![];
                                let mut srch = Search1::new(&p1, |path: &[u8], f| {
                                    eps.push((path.to_vec(), f));
                                    true
                                });
                                srch.pair = ALL_PAIRS;
                                srch.path.extend_from_slice(&q);
                                srch.dfs(&c, dq, len - 5, q[4] as usize);
                                let nodes = srch.nodes;
                                drop(srch);
                                let mut counts = [0u16; 6];
                                for (_, _, h2) in candidate_bounds(&p2, &cube, &eps, limit + 2) {
                                    let v = len as usize + h2 as usize;
                                    let k = (v + 3).saturating_sub(limit).min(5);
                                    counts[k] = counts[k].saturating_add(1);
                                    tc[k] += 1;
                                }
                                tn += nodes;
                                te += eps.len();
                                record(
                                    &mut buf,
                                    ci,
                                    len,
                                    &q,
                                    dq,
                                    nodes,
                                    eps.len(),
                                    &counts,
                                    features(&c, &low),
                                );
                            }
                        }
                        out.lock().unwrap().push(buf);
                        let mut t = totals.lock().unwrap();
                        t.0 += tn;
                        t.1 += te;
                        for k in 0..6 {
                            t.2[k] += tc[k];
                        }
                    });
                }
            });
            for b in out.into_inner().unwrap() {
                file.write_all(&b).unwrap();
            }
            let (tn, te, tc) = totals.into_inner().unwrap();
            println!(
                "cube {ci} d0 {d0} len {len}: {} jobs sampled, nodes {tn}, endpoints {te}, candidates by L1+h2 {}..{}: {tc:?}, {:.1} s",
                items.len(),
                limit - 3,
                limit + 2,
                t0.elapsed().as_secs_f64()
            );
            std::io::stdout().flush().unwrap();
        }
    }
}
