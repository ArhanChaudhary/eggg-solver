//! Finishing study: what the phase-2 + phase-3 split leaves on the table, measured on real
//! phase-1 endpoints.
//!
//! For candidates (phase-1 endpoints in a phase-2 frame) with phase-2 bound h2, every phase-2
//! solution (reduction) of length h2 + s, s = 0..=S, is enumerated with the solver's joint
//! search. With a remainder budget B (the moves left for phases 2 + 3 to beat the best
//! solution: B = best - 1 - L1), a reduction at slack s needs a phase 3 of at most
//! y = B - h2 - s moves. Per candidate and slack the program prints the reductions, the
//! phase-2 nodes and CPU time, the same search with a corner bound (a phase-2 node with
//! `left` moves to go is entered only if the corners, which every phase-2 move turns like a
//! 3x3x3 move, are within left + y outer moves of solved in one of the four orientations the
//! centres may end in), the reductions passing the H48 bound (<= y) and the corner bound, and
//! the chance that a reduction of the candidate has a phase 3 of at most y moves: the edge
//! orientation of the 3x3x3 after reduction (relative to the R/L axis, the one the phase-2
//! moves never flip) is fixed by phase 1 (which wing of each dedge is low), so the chance is
//! P(d3 <= y | that edge orientation), from the number of 3x3x3 states within y moves of solved
//! with each edge orientation (counted over canonical move sequences, scaled to the exact
//! state counts).
//!
//! Usage: finish CUBES ENDPOINTS_PER_CUBE LEN_OFFSET MAX_SLACK SEED THREADS B H2LO H2HI
//! PER_H2 [H48_K] (tables in $EGGG_TABLES or ./tables). Endpoints: per cube, phase-1 searches
//! of length d0 + LEN_OFFSET below random 3-move prefixes; candidates with h2 in H2LO..=H2HI,
//! at most PER_H2 per h2 value (spread over the cubes).
use eggg_solver::comb::perm_rank;
use eggg_solver::cube::{Cube, F, R, Turn, U};
use eggg_solver::cube3::{Cube3, MOVES3};
use eggg_solver::h48::{Syms, sym_mats};
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES, frame_rotation, rotate_move};
use eggg_solver::phase1::{P1State, Search1};
use eggg_solver::phase2::{P2_MOVES, Search2, first_moves, standardise};
use eggg_solver::phase3::P3Tables;
use eggg_solver::solver::{load_p1, load_p2};
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

fn cpu_now() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// Exact numbers of 3x3x3 states at each distance (half-turn metric), 0..=14.
const D3_COUNT: [f64; 15] = [
    1.0,
    18.0,
    243.0,
    3240.0,
    43239.0,
    574908.0,
    7618438.0,
    100803036.0,
    1332343288.0,
    17596479795.0,
    232248063316.0,
    3063288809012.0,
    40374425656248.0,
    531653418284628.0,
    6989320578825358.0,
];
const D3_TOTAL: f64 = 43252003274489856000.0;
const MAXY: usize = 14;

const N_CP: usize = 40320;
const N_CO: usize = 2187;
const N_CORNER: usize = N_CP * N_CO;

fn co_index(co: &[u8; 8]) -> usize {
    co[..7].iter().fold(0, |a, &x| a * 3 + x as usize)
}

fn co_unindex(mut i: usize) -> [u8; 8] {
    let mut co = [0u8; 8];
    let mut s = 0;
    for k in (0..7).rev() {
        co[k] = (i % 3) as u8;
        s += co[k];
        i /= 3;
    }
    co[7] = (3 - s % 3) % 3;
    co
}

fn cp_unrank(r: usize) -> [u8; 8] {
    let v = eggg_solver::comb::perm_unrank(r as u64, 8);
    std::array::from_fn(|i| v[i])
}

/// Corner coordinate tables: moves (the 18 outer moves, then the 17 phase-2 moves) and the
/// distance to solved in any of the orientations I, x2, y2, z2.
struct Corners {
    cp_move: Vec<[u16; 35]>,
    co_move: Vec<[u16; 35]>,
    dist: Vec<u8>,
}

impl Corners {
    fn index(cp: &[u8; 8], co: &[u8; 8]) -> u32 {
        (perm_rank(cp) as usize * N_CO + co_index(co)) as u32
    }

    /// `multi`: distance to the four end orientations (else to solved only).
    fn new(threads: usize, multi: bool) -> Corners {
        let moves: Vec<usize> = (0..18)
            .chain(P2_MOVES.iter().map(|&m| m as usize))
            .collect();
        let mut cp_move = vec![[0u16; 35]; N_CP];
        for (r, row) in cp_move.iter_mut().enumerate() {
            let cp = cp_unrank(r);
            for (k, &m) in moves.iter().enumerate() {
                let p = &MOVE_PERMS[m];
                let n: [u8; 8] = std::array::from_fn(|t| cp[p.corner_src[t] as usize]);
                row[k] = perm_rank(&n) as u16;
            }
        }
        let mut co_move = vec![[0u16; 35]; N_CO];
        for (r, row) in co_move.iter_mut().enumerate() {
            let co = co_unindex(r);
            for (k, &m) in moves.iter().enumerate() {
                let p = &MOVE_PERMS[m];
                let n: [u8; 8] =
                    std::array::from_fn(|t| (co[p.corner_src[t] as usize] + p.corner_twist[t]) % 3);
                row[k] = co_index(&n) as u16;
            }
        }
        let dist: Vec<AtomicU8> = (0..N_CORNER).map(|_| AtomicU8::new(255)).collect();
        for t in [
            None,
            Some(Turn::rotation(R, 2)),
            Some(Turn::rotation(U, 2)),
            Some(Turn::rotation(F, 2)),
        ]
        .into_iter()
        .take(if multi { 4 } else { 1 })
        {
            let c = t.map_or(Cube::SOLVED, |t| Cube::SOLVED.apply_turn(t));
            dist[Self::index(&c.cp, &c.co) as usize].store(0, Ordering::Relaxed);
        }
        let mut d = 0u8;
        loop {
            let next = AtomicUsize::new(0);
            let added = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        loop {
                            let b = next.fetch_add(1 << 16, Ordering::Relaxed);
                            if b >= N_CORNER {
                                break;
                            }
                            let mut n = 0;
                            for i in b..(b + (1 << 16)).min(N_CORNER) {
                                if dist[i].load(Ordering::Relaxed) != d {
                                    continue;
                                }
                                let (cp, co) = (i / N_CO, i % N_CO);
                                for k in 0..18 {
                                    let j =
                                        cp_move[cp][k] as usize * N_CO + co_move[co][k] as usize;
                                    if dist[j].load(Ordering::Relaxed) == 255 {
                                        dist[j].store(d + 1, Ordering::Relaxed);
                                        n += 1;
                                    }
                                }
                            }
                            added.fetch_add(n, Ordering::Relaxed);
                        }
                    });
                }
            });
            if added.load(Ordering::Relaxed) == 0 {
                break;
            }
            d += 1;
        }
        let dist = dist.into_iter().map(|a| a.into_inner()).collect();
        Corners {
            cp_move,
            co_move,
            dist,
        }
    }

    #[inline]
    fn step(&self, idx: u32, k: usize) -> u32 {
        let (cp, co) = (idx as usize / N_CO, idx as usize % N_CO);
        (self.cp_move[cp][k] as usize * N_CO + self.co_move[co][k] as usize) as u32
    }
}

/// Edge orientation statistics of the 3x3x3 ball: `p[y][pattern]` = P(d3 <= y | the edges
/// flipped (by piece) relative to the F/B axis are `pattern`).
fn eo_chances() -> Vec<Vec<f64>> {
    // seq[last face][pattern]: canonical sequences of the current length (pattern by position).
    let mut cur = vec![vec![0f64; 4096]; 7];
    cur[6][0] = 1.0;
    let mut cum = vec![0f64; 4096];
    cum[0] = 1.0;
    let mut out = vec![];
    let norm = |cum: &Vec<f64>, y: usize| -> Vec<f64> {
        let tot: f64 = cum.iter().sum();
        let exact: f64 = D3_COUNT[..=y].iter().sum();
        cum.iter()
            .map(|&c| c / tot * exact / (D3_TOTAL / 2048.0))
            .collect()
    };
    out.push(norm(&cum, 0));
    for y in 1..=MAXY {
        let mut nxt = vec![vec![0f64; 4096]; 7];
        for lf in 0..7 {
            for (pat, &v) in cur[lf].iter().enumerate() {
                if v == 0.0 {
                    continue;
                }
                for m in 0..18 {
                    let f = m / 3;
                    if lf < 6 && ALLOWED_AFTER[lf * 3] >> m & 1 == 0 {
                        continue;
                    }
                    let mv = &MOVES3[m];
                    let mut np = 0usize;
                    for t in 0..12 {
                        let b = (pat >> mv.edge_src[t] as usize & 1) as u8 ^ mv.edge_flip[t];
                        np |= (b as usize) << t;
                    }
                    nxt[f][np] += v;
                }
            }
        }
        cur = nxt;
        for row in &cur {
            for (c, &v) in cum.iter_mut().zip(row) {
                *c += v;
            }
        }
        out.push(norm(&cum, y));
    }
    out
}

struct Cand {
    cube: usize,
    len1: usize,
    frame: usize,
    x: Cube,
    cc: u32,
    tau: [u8; 16],
    dj: u8,
    df: u8,
    h2: u8,
    last1: usize,
    first: u32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: u64| -> u64 { args.get(i).map(|s| s.parse().unwrap()).unwrap_or(d) };
    let ncubes = arg(1, 4) as usize;
    let per_cube = arg(2, 20000) as usize;
    let off = arg(3, 4) as u8;
    let max_slack = arg(4, 2) as u8;
    let seed = arg(5, 1);
    let threads = arg(6, 4).max(1) as usize;
    let budget = arg(7, 20) as u8;
    let (h2lo, h2hi) = (arg(8, 9) as u8, arg(9, 11) as u8);
    let per_h2 = arg(10, 50) as usize;
    let h48k = arg(11, 6) as u32;
    let dir = std::env::var_os("EGGG_TABLES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "tables".into());
    let t0 = std::time::Instant::now();
    let p1 = load_p1(&dir, false, threads, false);
    let p2 = &load_p2(&dir, true, threads, false);
    let p3 = &P3Tables::load_or_build(&dir, h48k, threads, false);
    let j = p2.joint.as_ref().unwrap();
    eprintln!("tables {:.1} s", t0.elapsed().as_secs_f64());
    let corners = &Corners::new(threads, true);
    // Check mode: the single-orientation table, to test the corner bound against the 3x3x3
    // corners of each reduction.
    let check = std::env::var_os("FINISH_CHECK").is_some();
    let single = check.then(|| Corners::new(threads, false));
    let mut hist = [0u64; 16];
    for &d in &corners.dist {
        hist[d.min(15) as usize] += 1;
    }
    eprintln!(
        "corner table {:.1} s, distances {:?}",
        t0.elapsed().as_secs_f64(),
        hist
    );
    let eo = &eo_chances();
    eprintln!("eo chances {:.1} s", t0.elapsed().as_secs_f64());
    // The symmetry taking the R/L axis to the F/B axis (fixing the U/D axis).
    let mats = sym_mats();
    let ksym = (0..48)
        .find(|&k| {
            let m = &mats[k];
            eggg_solver::cube::mat_apply(m, [1, 0, 0])[2].abs() == 1
                && eggg_solver::cube::mat_apply(m, [0, 1, 0]) == [0, 1, 0]
        })
        .unwrap();
    let syms = &Syms::new();
    // Pattern of flipped edge pieces relative to the R/L axis.
    let eo_rl = |c3: &Cube3| -> usize {
        let y = syms.conj(c3, ksym);
        let mut pat = 0usize;
        for i in 0..12 {
            if y.eo[i] == 1 {
                pat |= 1 << y.ep[i];
            }
        }
        pat
    };
    // Check: per piece, the R/L edge orientation is kept by U D F B quarter turns and R2 L2,
    // and R and L quarter turns flip four pieces.
    {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
        for _ in 0..200 {
            let mut c3 = Cube3::SOLVED;
            for _ in 0..30 {
                c3 = c3.apply(rng.gen_range(0..18));
            }
            let e0 = eo_rl(&c3);
            for m in 0..18 {
                let e1 = eo_rl(&c3.apply(m));
                let quarter_rl = (m / 3 == 1 || m / 3 == 4) && m % 3 != 1;
                assert_eq!(e0 != e1, quarter_rl, "move {m}");
                if quarter_rl {
                    assert_eq!((e0 ^ e1).count_ones(), 4);
                }
            }
        }
    }
    // Endpoints and candidates.
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut cands: Vec<Cand> = vec![];
    let mut h2_hist = [0u64; 32];
    for ci in 0..ncubes {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        let len = d0 + off;
        let mut eps: Vec<(Vec<u8>, u8)> = vec![];
        // The canonical 2-move prefixes within reach, in random order; up to 1/64 of the
        // endpoints below each (so the sample spreads over the tree).
        let mut pres = vec![];
        for a in 0..N_MOVES {
            for b in 0..N_MOVES {
                if ALLOWED_AFTER[a] >> b & 1 == 0 {
                    continue;
                }
                let s = root.apply(a, &p1).apply(b, &p1);
                let d = p1.distance(&s);
                if d <= len - 2 {
                    pres.push((vec![a as u8, b as u8], s, d, b));
                }
            }
        }
        use rand::seq::SliceRandom;
        pres.shuffle(&mut rng);
        for (pre, s, d, last) in &pres {
            if eps.len() >= per_cube {
                break;
            }
            let want = (per_cube - eps.len()).min(per_cube / 64 + 1);
            let mut got = 0;
            let mut srch = Search1::new(&p1, |p: &[u8], f| {
                let mut v = pre.clone();
                v.extend_from_slice(p);
                eps.push((v, f));
                got += 1;
                got < want
            });
            srch.dfs(s, *d, len - 2, *last);
        }
        for (moves, frames) in &eps {
            let mut c = cube;
            for &m in moves {
                c = c.apply_perm(&MOVE_PERMS[m as usize]);
            }
            for f in 0..3 {
                if frames >> f & 1 == 0 {
                    continue;
                }
                let x = standardise(&c, f);
                let cc = p2.centre_coord_of(&x);
                let cd = p2.centre_dist[cc as usize];
                if cd == eggg_solver::table::UNSEEN {
                    continue;
                }
                let tau = p2.tau_of(&x);
                let dj = j.distance(cc / 1680, &tau, cc & 1);
                let df = j.distance_fb(cc, &tau);
                let h2 = cd.max(dj).max(df);
                h2_hist[h2 as usize] += 1;
                if h2 < h2lo || h2 > h2hi {
                    continue;
                }
                let rot = frame_rotation(f);
                let last1 = moves
                    .last()
                    .map(|&m| rotate_move(m as usize, &rot))
                    .unwrap_or(N_MOVES);
                let tail: Vec<u8> = moves[moves.len().saturating_sub(3)..]
                    .iter()
                    .map(|&m| rotate_move(m as usize, &rot) as u8)
                    .collect();
                cands.push(Cand {
                    cube: ci,
                    len1: moves.len(),
                    frame: f,
                    x,
                    cc,
                    tau,
                    dj,
                    df,
                    h2,
                    last1,
                    first: first_moves(&tail),
                });
            }
        }
        eprintln!(
            "cube {ci}: d0 {d0}, {} endpoints of length {len}, {:.1} s",
            eps.len(),
            t0.elapsed().as_secs_f64()
        );
    }
    println!(
        "# h2 histogram (all frames of all endpoints): {:?}",
        &h2_hist[..20]
    );
    // At most per_h2 candidates per h2 value, round robin over the cubes.
    let mut sel: Vec<usize> = vec![];
    for h in h2lo..=h2hi {
        let mut by_cube: Vec<Vec<usize>> = vec![vec![]; ncubes];
        for (i, c) in cands.iter().enumerate() {
            if c.h2 == h {
                by_cube[c.cube].push(i);
            }
        }
        let mut k = 0;
        let mut taken = 0;
        while taken < per_h2 && by_cube.iter().any(|v| k < v.len()) {
            for v in &by_cube {
                if taken < per_h2 && k < v.len() {
                    sel.push(v[k]);
                    taken += 1;
                }
            }
            k += 1;
        }
    }
    eprintln!("{} candidates selected", sel.len());
    // Per candidate: one line per slack.
    let next = AtomicUsize::new(0);
    let lines = std::sync::Mutex::new(vec![]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= sel.len() {
                        break;
                    }
                    let c = &cands[sel[k]];
                    let cidx = Corners::index(&c.x.cp, &c.x.co);
                    let mut pats = std::collections::BTreeSet::new();
                    let mut out = vec![];
                    for sl in 0..=max_slack {
                        let l2 = c.h2 + sl;
                        let y = budget as i32 - l2 as i32;
                        // Plain search (as the solver), with per-reduction statistics.
                        let t = cpu_now();
                        let mut srch = Search2::new(p2);
                        let (mut n, mut n_h3, mut n_cd) = (0u64, 0u64, 0u64);
                        let mut chance = 0f64;
                        srch.dfs_joint(c.cc, &c.tau, c.dj, c.df, l2, c.last1, c.first, &mut |p| {
                            n += 1;
                            let mut x = c.x;
                            for &m in p {
                                x = x.apply_perm(&MOVE_PERMS[m as usize]);
                            }
                            let c3 = Cube3::from_reduced(&x).expect("not reduced");
                            if let Some(d1) = &single {
                                let a = corners.dist[Corners::index(&x.cp, &x.co) as usize];
                                let b = d1.dist[Corners::index(&c3.cp, &c3.co) as usize];
                                assert!(a <= b, "corner bound {a} > 3x3x3 corner distance {b}");
                            }
                            let pat = eo_rl(&c3);
                            pats.insert(pat.count_ones() * 10000 + (eo[8][pat] * 1e6) as u32);
                            if y >= 0 {
                                let yy = (y as usize).min(MAXY);
                                chance += eo[yy][pat];
                                if p3.lower_bound(&c3) as i32 <= y {
                                    n_h3 += 1;
                                }
                                if corners.dist[Corners::index(&x.cp, &x.co) as usize] as i32 <= y {
                                    n_cd += 1;
                                }
                            }
                            true
                        });
                        let (nodes, cpu) = (srch.nodes, cpu_now() - t);
                        // The same with the corner bound.
                        let (mut pn, mut pnodes, mut pcpu) = (0u64, 0u64, 0f64);
                        let dbg = std::env::var_os("FINISH_LEVELS").is_some();
                        let lv = std::sync::Mutex::new([[0u64; 2]; 32]);
                        if y >= 0 {
                            let t = cpu_now();
                            let mut srch = Search2::new(p2);
                            srch.dfs_joint_side(
                                c.cc,
                                &c.tau,
                                c.dj,
                                c.df,
                                l2,
                                c.last1,
                                c.first,
                                cidx,
                                &|&ci: &u32, i| corners.step(ci, 18 + i),
                                &|&ci: &u32, left| {
                                    let ok = corners.dist[ci as usize] as i32 <= left as i32 + y;
                                    if dbg {
                                        let mut v = lv.lock().unwrap();
                                        v[left as usize][ok as usize] += 1;
                                    }
                                    ok
                                },
                                &mut |_, _| {
                                    pn += 1;
                                    true
                                },
                            );
                            pnodes = srch.nodes;
                            pcpu = cpu_now() - t;
                            // The corner bound cuts exactly the reductions whose corners are
                            // too far (it is consistent under the phase-2 moves).
                            assert_eq!(pn, n_cd, "corner-bound search lost reductions");
                        }
                        if dbg {
                            let v = lv.lock().unwrap();
                            let w: Vec<String> = (0..=l2 as usize).filter(|&l| v[l][0] + v[l][1] > 0).map(|l| format!("{l}:{}/{}", v[l][1], v[l][0] + v[l][1])).collect();
                            out.push(format!("#   levels (left: kept/joint-ok) {}", w.join(" ")));
                        }
                        out.push(format!(
                            "{k} {} {} {} {} {} {} {sl} {y} {n} {nodes} {cpu:.6} {pn} {pnodes} {pcpu:.6} {n_h3} {n_cd} {chance:.6e}",
                            c.cube, c.len1, c.frame, c.h2, c.dj, c.df
                        ));
                    }
                    // The edge orientation must be the same for all reductions of a candidate
                    // (counted by flips and chance; the four end orientations relabel pieces).
                    let line = format!("{} PATS {:?}", out.join("\n"), pats);
                    lines.lock().unwrap().push((k, line));
                }
            });
        }
    });
    let mut lines = lines.into_inner().unwrap();
    lines.sort();
    println!(
        "# k cube len1 frame h2 dj df slack y reds nodes cpu reds_cp nodes_cp cpu_cp reds_h3ok reds_cdok chance_sum"
    );
    for (_, l) in lines {
        println!("{l}");
    }
    eprintln!("done {:.1} s", t0.elapsed().as_secs_f64());
}
