//! Experiment: endings "... X W" (X outer moves, W a wide half turn) after a phase-2 solution
//! with its last move removed.
//!
//! A phase-2 solution R = Q W (W its final wide half turn) reaches the reduced cube r; the
//! solver finishes with an optimal 3x3x3 solution of r (d3(r) moves). From the cube after Q,
//! the alternative "X W'" needs the cube after Q X to be the solved cube with W'^-1 applied
//! (in some orientation). Outer moves keep the centres, so the centre pattern forces W' = W;
//! wings and corners are distinguishable, so X must take the wings and corners of the cube
//! after Q, q, to those of W: X solves p = W * q (W applied first, then q's permutation), if p
//! is a reduced (paired) state at all. This program measures, for each reduced r and each W:
//! whether such an X exists, its length against d3(r), and verifies every ending found on the
//! facelets.
//!
//! r is either a random 3x3x3 state on a reduced 4x4x4 (random outer sequences), or read from
//! a dump of real reductions (lines "FACELETS MOVES12", MOVES12 as move indices, as written by
//! the solver with EGGG_DUMP_REDS=file).
//!
//! usage: xwend random N [THREADS] [H48_K] | xwend dump FILE [THREADS] [H48_K]

use eggg_solver::cube::{Cube, Facelets};
use eggg_solver::cube3::Cube3;
use eggg_solver::moves::MOVE_PERMS;
use eggg_solver::phase3::{P3Tables, Search3};
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const WIDE_HALF: [usize; 3] = [19, 22, 25];

/// The composition "a then b" of two labelled states (centres are taken from `b` relabelled
/// by `a`, which is meaningless for centres; callers overwrite them).
fn compose(a: &Cube, b: &Cube) -> Cube {
    let mut c = *b;
    for t in 0..8 {
        let s = b.cp[t] as usize;
        c.cp[t] = a.cp[s];
        c.co[t] = (a.co[s] + b.co[t]) % 3;
    }
    for t in 0..24 {
        c.wp[t] = a.wp[b.wp[t] as usize];
        c.ct[t] = a.ct[b.ct[t] as usize];
    }
    c
}

fn apply(c: &Cube, ms: &[usize]) -> Cube {
    ms.iter().fold(*c, |x, &m| x.apply_perm(&MOVE_PERMS[m]))
}

/// Recolour `x` so that the centres of `r` (reduced, same colouring) are in their home
/// orientation; returns the recoloured x.
fn home(r: &Cube, x: &Cube) -> Option<Cube> {
    let cols = r.centre_colours();
    let face_colour: Vec<u8> = (0..6).map(|f| cols[f * 4]).collect();
    eggg_solver::cube::ROTATIONS
        .iter()
        .map(|m| eggg_solver::moves::Conj::new(*m))
        .find(|cj| {
            let map = cj.colour_map();
            (0..6).all(|f| map[face_colour[f] as usize] as usize == f)
        })
        .map(|cj| cj.recolour(x))
}

struct Res {
    d3: u8,
    /// Per W: (a lower bound on |X|, whether it is exact), None: no ending X W.
    x: [Option<(u8, bool)>; 3],
}

/// For each wide half turn W', the shortest X with r W' X W' solved, if any. `q` is the cube
/// after Q (the reduction without its last move `w_last`): q = r W, so for W' = W this is the
/// alternative ending "X W" of Q; for the other W' it replaces phase 3 by "W' X W'".
fn evaluate(s: &mut Search3, q: &Cube, w_last: usize, r: &Cube) -> Res {
    let r3 = Cube3::from_reduced(r).expect("not reduced");
    let mut x = [None; 3];
    // Work in the colouring where r's centres are at home (recolouring commutes with moves).
    let (qh, rh) = (home(r, q).unwrap(), home(r, r).unwrap());
    for (i, &w) in WIDE_HALF.iter().enumerate() {
        let qw = apply(&rh, &[w]);
        if w == w_last {
            assert_eq!(qw, qh);
        }
        let wcube = apply(&Cube::SOLVED, &[w]);
        let mut p = compose(&wcube, &qw);
        p.ct = Cube::SOLVED.ct;
        if let Some(p3) = Cube3::from_reduced(&p) {
            // X must solve p3 and also turn each centre block off W's axis a multiple of four
            // quarter turns (those blocks are two-coloured before W). The optimal X of p3
            // without that condition bounds |X| from below; it is exact when it happens to
            // meet the condition (checked on the facelets).
            let sol = s.solve(&p3, 20).unwrap();
            let mut v: Vec<usize> = sol.iter().map(|&m| m as usize).collect();
            v.push(w);
            let ok = apply(&qw, &v).to_facelets().is_solved();
            x[i] = Some((sol.len() as u8, ok));
        }
    }
    // The optimal phase 3 (slow for random cubes) only matters where an ending exists.
    let d3 = if x.iter().any(|v| v.is_some()) {
        s.solve(&r3, 20).unwrap().len() as u8
    } else {
        0
    };
    Res { d3, x }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("random");
    let threads: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(6);
    let k: u32 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(5);
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), k, threads, true);

    // Inputs: (cube after Q, last move W, reduction r).
    let inputs: Vec<(Cube, usize, Cube)> = match mode {
        "random" => {
            let n: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(10000);
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(11);
            (0..n)
                .map(|_| {
                    let ms: Vec<usize> = (0..40).map(|_| rng.gen_range(0..18)).collect();
                    let r = apply(&Cube::SOLVED, &ms);
                    let w = WIDE_HALF[rng.gen_range(0..3)];
                    (apply(&r, &[w]), w, r)
                })
                .collect()
        }
        _ => {
            let text = std::fs::read_to_string(&args[2]).expect("reading dump");
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let mut it = l.split_whitespace();
                    let f = Facelets::parse(it.next().unwrap()).unwrap();
                    let (cube, _) = f.to_cube().unwrap();
                    let ms: Vec<usize> = it.map(|m| m.parse().unwrap()).collect();
                    let w = *ms.last().unwrap();
                    assert!(
                        WIDE_HALF.contains(&w),
                        "reduction must end with a wide half turn"
                    );
                    let q = apply(&cube, &ms[..ms.len() - 1]);
                    let r = apply(&q, &[w]);
                    (q, w, r)
                })
                .collect()
        }
    };
    let next = AtomicUsize::new(0);
    let out = Mutex::new(vec![]);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut s = Search3::new(&t);
                let mut local = vec![];
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= inputs.len() {
                        break;
                    }
                    let (q, w, r) = &inputs[i];
                    local.push((i, evaluate(&mut s, q, *w, r)));
                }
                out.lock().unwrap().extend(local);
            });
        }
    });
    let res = out.into_inner().unwrap();
    let n = res.len();
    // For the actual last move W of each reduction: X W endings and their gain d3 - |X|.
    let mut exists = 0;
    let mut gain: BTreeMap<(i32, &str), usize> = BTreeMap::new();
    let mut any_w = 0;
    let mut gain_any: BTreeMap<(i32, &str), usize> = BTreeMap::new();
    let mut d3h: BTreeMap<u8, usize> = BTreeMap::new();
    // Gains are upper bounds where the optimal X misses the centre condition ("bound").
    let tag = |ok: bool| if ok { "exact" } else { "bound" };
    for (i, r) in &res {
        *d3h.entry(r.d3).or_default() += 1;
        let wi = WIDE_HALF.iter().position(|&w| w == inputs[*i].1).unwrap();
        if let Some((x, ok)) = r.x[wi] {
            exists += 1;
            *gain.entry((r.d3 as i32 - x as i32, tag(ok))).or_default() += 1;
        }
        if let Some((x, ok)) = r.x.iter().flatten().min() {
            any_w += 1;
            *gain_any
                .entry((r.d3 as i32 - *x as i32 - 2, tag(*ok)))
                .or_default() += 1;
        }
    }
    println!("{n} reductions; d3 histogram (0: not computed, no ending) {d3h:?}");
    println!(
        "ending Q X W (same W): wings and corners allow it for {exists}; gain d3(r) - |X| (Q X W vs Q W + phase 3): {gain:?}"
    );
    println!(
        "phase 3 replaced by W' X W' (any W'): allowed for {any_w}; gain d3(r) - |X| - 2: {gain_any:?}"
    );
}
