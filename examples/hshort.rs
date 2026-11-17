//! Experiment: can a last phase in H = <U R F D L B, Uw2 Rw2 Fw2> beat what the solver already
//! covers on reduced cubes?
//!
//! Enumerates every canonical H-sequence from the solved cube up to `MAXLEN` moves, keeps
//! those that contain a wide half turn and end in a reduced cube (centres solved in some
//! orientation, dedges paired), and compares, per reduced state z:
//! - `kH`: its shortest H-sequence (the inverse solves z in H),
//! - `d3`: its optimal 3x3x3 distance (outer moves, HTM),
//! - `ka` (a = x, y, z axis): its shortest sequence whose inverse has the shape the solver
//!   already enumerates in the frame where axis a is the half-turn axis of phase 2: phase-2
//!   moves (no quarter turns of axis a) up to the last wide move, then any outer moves. Such
//!   a solving tail is just a longer phase-2 solution plus phase 3.
//!
//! A state with d3 > kH is a shortcut over phase 3; one with min(d3, ka) > kH is a shortcut
//! the solver cannot produce in frame a. Reduced states with PLL parity are counted apart.
//!
//! usage: hshort MAXLEN [THREADS] [H48_K]

use eggg_solver::cube::Cube;
use eggg_solver::cube3::Cube3;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES, move_of};
use eggg_solver::notation::{Style, format_moves};
use eggg_solver::phase3::{P3Tables, Search3};
use eggg_solver::simd::V32;
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const OUTER: u32 = (1 << 18) - 1;
const WIDE2: u32 = (1 << 19) | (1 << 22) | (1 << 25);
const HMASK: u32 = OUTER | WIDE2;
const NONE: u8 = 99;

#[derive(Clone, Copy)]
struct St {
    w: V32,
    c: V32,
}

struct Ctx {
    wsrc: Vec<V32>,
    csrc: Vec<V32>,
}

/// Per reduced state: [kH, kx, ky, kz] (NONE: not found), and a sequence of length kH.
type Rec = ([u8; 4], Vec<u8>);

#[derive(Default)]
struct Found {
    /// Reduced (legal 3x3x3) states.
    red: HashMap<Cube3, Rec>,
    /// Reduced states with PLL parity.
    par: HashMap<Cube3, Rec>,
    /// Sequences (with a wide move) ending reduced, per length.
    seqs: [u64; 16],
    seqs_par: [u64; 16],
    nodes: u64,
}

fn centres_uniform(c: &V32) -> bool {
    (0..6).all(|f| {
        let x = u32::from_le_bytes(c.0[f * 4..f * 4 + 4].try_into().unwrap());
        x == (x & 255).wrapping_mul(0x0101_0101)
    })
}

fn wings_paired(w: &V32) -> bool {
    (0..12).all(|e| w.0[2 * e] >> 1 == w.0[2 * e + 1] >> 1)
}

fn merge(map: &mut HashMap<Cube3, Rec>, x: Cube3, ls: [u8; 4], path: &[u8]) {
    let e = map.entry(x).or_insert(([NONE; 4], vec![]));
    if ls[0] < e.0[0] {
        e.1 = path.to_vec();
    }
    for i in 0..4 {
        e.0[i] = e.0[i].min(ls[i]);
    }
}

fn record(map: &mut HashMap<Cube3, Rec>, x: Cube3, path: &[u8], bad: u8) {
    let len = path.len() as u8;
    let mut ls = [len; 4];
    for a in 0..3 {
        if bad >> a & 1 == 1 {
            ls[a + 1] = NONE;
        }
    }
    merge(map, x, ls, path);
}

/// Update (wide moves so far, axes with a quarter turn after a wide move) for move `m`. In the
/// solving (inverse) sequence such a quarter turn comes before a wide move, which phase 2 in
/// the frame of that axis cannot do.
fn shape(m: usize, wides: u32, bad: u8) -> (u32, u8) {
    if WIDE2 >> m & 1 == 1 {
        return (wides + 1, bad);
    }
    if m < 18 && m % 3 != 1 && wides > 0 {
        // Faces U R F D L B: axes y x z y x z; bit 0 = x, 1 = y, 2 = z.
        let axis = [1, 0, 2][(m / 3) % 3];
        return (wides, bad | 1 << axis);
    }
    (wides, bad)
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    ctx: &Ctx,
    st: &St,
    left: u32,
    last: usize,
    wides: u32,
    bad: u8,
    path: &mut Vec<u8>,
    f: &mut Found,
    only_short: &mut Option<Search3>,
) {
    f.nodes += 1;
    if wides > 0 && centres_uniform(&st.c) && wings_paired(&st.w) {
        let cube = path
            .iter()
            .fold(Cube::SOLVED, |c, &m| c.apply_perm(&MOVE_PERMS[m as usize]));
        if let Some(x) = Cube3::from_reduced_any(&cube) {
            if x.is_valid() {
                f.seqs[path.len()] += 1;
                // With `only_short`, keep only the states that are shortcuts at this length.
                let keep = match only_short {
                    Some(s) => s.solve(&x, path.len() as u8).is_none(),
                    None => true,
                };
                if keep {
                    record(&mut f.red, x, path, bad);
                }
            } else {
                f.seqs_par[path.len()] += 1;
                record(&mut f.par, x, path, bad);
            }
        }
    }
    if left == 0 {
        return;
    }
    let mut allowed = ALLOWED_AFTER[last] & HMASK;
    while allowed != 0 {
        let m = allowed.trailing_zeros() as usize;
        allowed &= allowed - 1;
        let n = St {
            w: st.w.gather(&ctx.wsrc[m]),
            c: st.c.gather(&ctx.csrc[m]),
        };
        path.push(m as u8);
        let (w2, b2) = shape(m, wides, bad);
        dfs(ctx, &n, left - 1, m, w2, b2, path, f, only_short);
        path.pop();
    }
}

fn seq_string(p: &[u8]) -> String {
    let ms: Vec<_> = p.iter().map(|&m| move_of(m as usize)).collect();
    format_moves(&ms, Style::Wca)
}

fn inverse_seq(p: &[u8]) -> Vec<u8> {
    p.iter()
        .rev()
        .map(|&m| eggg_solver::moves::inverse_move(m as usize) as u8)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let maxlen: u32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(7);
    let threads: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(6);
    let k: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);
    // "short": keep only the shortcut states during the enumeration (little memory, no d3
    // histogram of the other states).
    let only_short = args.get(4).is_some_and(|s| s == "short");
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), k, threads, true);

    let mk = |src: &[u8; 24]| {
        let mut v = [0u8; 32];
        for (i, x) in v.iter_mut().enumerate() {
            *x = if i < 24 { src[i] } else { i as u8 };
        }
        V32(v)
    };
    let ctx = Ctx {
        wsrc: (0..N_MOVES).map(|m| mk(&MOVE_PERMS[m].wing_src)).collect(),
        csrc: (0..N_MOVES)
            .map(|m| mk(&MOVE_PERMS[m].centre_src))
            .collect(),
    };
    let solved = St {
        w: mk(&Cube::SOLVED.wp),
        c: mk(&Cube::SOLVED.centre_colours()),
    };

    // Tasks: canonical prefixes of length 2 (the shorter sequences are checked here).
    let mut tasks = vec![];
    let mut total = Found::default();
    {
        let mut path = vec![];
        let mut srch = only_short.then(|| Search3::new(&t));
        dfs(
            &ctx,
            &solved,
            1.min(maxlen),
            N_MOVES,
            0,
            0,
            &mut path,
            &mut total,
            &mut srch,
        );
    }
    let mut a = HMASK;
    while a != 0 && maxlen >= 2 {
        let m1 = a.trailing_zeros() as usize;
        a &= a - 1;
        let mut b = ALLOWED_AFTER[m1] & HMASK;
        while b != 0 {
            let m2 = b.trailing_zeros() as usize;
            b &= b - 1;
            tasks.push((m1, m2));
        }
    }
    let t0 = std::time::Instant::now();
    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(0);
    let results = Mutex::new(vec![]);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut f = Found::default();
                let mut srch = only_short.then(|| Search3::new(&t));
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= tasks.len() {
                        break;
                    }
                    let (m1, m2) = tasks[i];
                    let s1 = St {
                        w: solved.w.gather(&ctx.wsrc[m1]),
                        c: solved.c.gather(&ctx.csrc[m1]),
                    };
                    let s2 = St {
                        w: s1.w.gather(&ctx.wsrc[m2]),
                        c: s1.c.gather(&ctx.csrc[m2]),
                    };
                    let (w1, b1) = shape(m1, 0, 0);
                    let (w2, b2) = shape(m2, w1, b1);
                    let mut path = vec![m1 as u8, m2 as u8];
                    dfs(
                        &ctx,
                        &s2,
                        maxlen - 2,
                        m2,
                        w2,
                        b2,
                        &mut path,
                        &mut f,
                        &mut srch,
                    );
                    done.fetch_add(1, Ordering::Relaxed);
                }
                results.lock().unwrap().push(f);
            });
        }
    });
    for f in results.into_inner().unwrap() {
        total.nodes += f.nodes;
        for i in 0..16 {
            total.seqs[i] += f.seqs[i];
            total.seqs_par[i] += f.seqs_par[i];
        }
        for (x, (ls, p)) in f.red {
            merge(&mut total.red, x, ls, &p);
        }
        for (x, (ls, p)) in f.par {
            merge(&mut total.par, x, ls, &p);
        }
    }
    eprintln!(
        "enumerated {} sequences up to length {maxlen} in {:.1} s",
        total.nodes,
        t0.elapsed().as_secs_f64()
    );

    // Optimal 3x3x3 distances of the reduced states.
    let states: Vec<(Cube3, [u8; 4], Vec<u8>)> = total
        .red
        .iter()
        .map(|(x, (ls, p))| (*x, *ls, p.clone()))
        .collect();
    let d3s: Vec<u8> = {
        let out = Mutex::new(vec![0u8; states.len()]);
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut srch = Search3::new(&t);
                    let mut local = vec![];
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= states.len() {
                            break;
                        }
                        let d = srch.solve(&states[i].0, 20).map(|s| s.len() as u8);
                        local.push((i, d.unwrap_or(NONE)));
                    }
                    let mut o = out.lock().unwrap();
                    for (i, d) in local {
                        o[i] = d;
                    }
                });
            }
        });
        out.into_inner().unwrap()
    };

    println!(
        "sequences (with a wide move) ending reduced, per length: {:?}",
        &total.seqs[..=maxlen as usize]
    );
    println!(
        "kH | distinct states | d3 histogram | d3 > kH (gain: count) | min(d3, kx) > kH | min(d3, kx, ky, kz) > kH"
    );
    let mut by_len: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
    for (i, s) in states.iter().enumerate() {
        by_len.entry(s.1[0]).or_default().push(i);
    }
    let mut examples = vec![];
    for (l, idx) in &by_len {
        let mut hist: BTreeMap<u8, u64> = BTreeMap::new();
        let mut gain3: BTreeMap<u8, u64> = BTreeMap::new();
        let mut gainx: BTreeMap<u8, u64> = BTreeMap::new();
        let mut gainall: BTreeMap<u8, u64> = BTreeMap::new();
        for &i in idx {
            let (_, ls, _) = &states[i];
            let d = d3s[i];
            *hist.entry(d).or_default() += 1;
            if d > *l {
                *gain3.entry(d - l).or_default() += 1;
            }
            let bx = d.min(ls[1]);
            if bx > *l {
                *gainx.entry(bx - l).or_default() += 1;
            }
            let ball = d.min(ls[1]).min(ls[2]).min(ls[3]);
            if ball > *l {
                *gainall.entry(ball - l).or_default() += 1;
                if examples.len() < 40 {
                    examples.push(i);
                }
            }
        }
        println!(
            "{l} | {} | {hist:?} | {gain3:?} | {gainx:?} | {gainall:?}",
            idx.len()
        );
    }
    println!("examples of shortcuts no frame covers (solving sequence = inverse):");
    for &i in &examples {
        let (x, ls, p) = &states[i];
        let sol = Search3::new(&t).solve(x, 20).unwrap();
        println!(
            "  H {} [{}] vs 3x3 {} [{}]; kx ky kz {:?}",
            ls[0],
            seq_string(&inverse_seq(p)),
            d3s[i],
            seq_string(&sol),
            &ls[1..]
        );
    }
    let mut pcv: BTreeMap<u8, u64> = BTreeMap::new();
    for (ls, _) in total.par.values() {
        *pcv.entry(ls[0]).or_default() += 1;
    }
    println!(
        "PLL-parity reduced states (distinct, by min H-length): {:?}; sequences per length {:?}",
        pcv,
        &total.seqs_par[..=maxlen as usize]
    );
    if let Some((ls, p)) = total.par.values().min_by_key(|(ls, _)| ls[0]) {
        println!(
            "shortest PLL parity in H: {} (shape lengths {:?})",
            seq_string(&inverse_seq(p)),
            ls
        );
    }
}
