//! Experiment: how often does a reduced cube at 3x3x3 distance d have a shorter solution in
//! H = <U R F D L B, Uw2 Rw2 Fw2>? (d up to ~13, by meet in the middle.)
//!
//! The states within `B` H-moves of the solved cube are stored (hashed, up to the final
//! orientation: every state is relabelled by the rotation that puts piece DBL untwisted into
//! slot DBL, and centres are compared by colour). For each test cube r (random outer
//! sequences, optimal 3x3x3 distance d computed), the states within d - 1 - B moves of r are
//! looked up: a hit is an H-solution of at most d - 1 moves. Hits are verified on the
//! facelets. Each solution found is also classified by shape: per axis a, whether it is a
//! phase-2 tail of the frame where axis a turns only by half turns (no quarter turn of axis a
//! before its last wide move), i.e. something the solver can already produce in that frame.
//!
//! usage: hmitm B SAMPLES_PER_D DMIN DMAX [THREADS] [H48_K]

use eggg_solver::cube::{Cube, DBL, ROTATIONS};
use eggg_solver::cube3::Cube3;
use eggg_solver::moves::{ALLOWED_AFTER, Conj, MOVE_PERMS, N_MOVES, inverse_move, move_of};
use eggg_solver::notation::{Style, format_moves};
use eggg_solver::phase3::{P3Tables, Search3};
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const OUTER: u32 = (1 << 18) - 1;
const WIDE2: u32 = (1 << 19) | (1 << 22) | (1 << 25);
const HMASK: u32 = OUTER | WIDE2;

#[derive(Default)]
struct Id(u64);
impl Hasher for Id {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
}
type Map = HashMap<u64, u64, BuildHasherDefault<Id>>;

struct Canon {
    /// For (piece in slot DBL, its twist): the relabelling that makes it DBL, untwisted.
    conj: Vec<Conj>,
    idx: [[usize; 3]; 8],
}

impl Canon {
    fn new() -> Canon {
        let conj: Vec<Conj> = ROTATIONS.iter().map(|m| Conj::new(*m)).collect();
        let mut idx = [[usize::MAX; 3]; 8];
        for (i, c) in conj.iter().enumerate() {
            for p in 0..8u8 {
                for o in 0..3u8 {
                    let mut x = Cube::SOLVED;
                    x.cp[DBL] = p;
                    x.co[DBL] = o;
                    let y = c.recolour(&x);
                    if y.cp[DBL] as usize == DBL && y.co[DBL] == 0 {
                        assert_eq!(idx[p as usize][o as usize], usize::MAX);
                        idx[p as usize][o as usize] = i;
                    }
                }
            }
        }
        assert!(idx.iter().flatten().all(|&i| i != usize::MAX));
        Canon { conj, idx }
    }

    fn key(&self, x: &Cube) -> u64 {
        let c = &self.conj[self.idx[x.cp[DBL] as usize][x.co[DBL] as usize]];
        let y = c.recolour(x);
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            h ^= h >> 31;
        };
        for i in 0..8 {
            mix((y.cp[i] * 3 + y.co[i]) as u64 + ((i as u64) << 8));
        }
        for i in 0..24 {
            mix(y.wp[i] as u64 + ((i as u64 + 8) << 8));
            mix((y.ct[i] / 4) as u64 + ((i as u64 + 40) << 8));
        }
        h
    }
}

fn encode(path: &[u8]) -> u64 {
    path.iter()
        .fold(path.len() as u64, |a, &m| a << 5 | m as u64)
}

fn decode(mut e: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    for i in (0..len).rev() {
        out[i] = (e & 31) as u8;
        e >>= 5;
    }
    out
}

fn len_of(e: u64, max: usize) -> usize {
    // The length is stored in the top bits: find it by matching.
    (0..=max).find(|&l| e >> (5 * l) == l as u64).unwrap()
}

/// All states within `depth` H-moves of `x`: callback with the state and the path.
fn walk(x: &Cube, depth: usize, last: usize, path: &mut Vec<u8>, f: &mut dyn FnMut(&Cube, &[u8])) {
    f(x, path);
    if depth == 0 {
        return;
    }
    let mut allowed = ALLOWED_AFTER[last] & HMASK;
    while allowed != 0 {
        let m = allowed.trailing_zeros() as usize;
        allowed &= allowed - 1;
        let y = x.apply_perm(&MOVE_PERMS[m]);
        path.push(m as u8);
        walk(&y, depth - 1, m, path, f);
        path.pop();
    }
}

fn apply(c: &Cube, ms: &[u8]) -> Cube {
    ms.iter()
        .fold(*c, |x, &m| x.apply_perm(&MOVE_PERMS[m as usize]))
}

/// Per axis (x, y, z): the solving sequence is a phase-2 tail of that axis's frame.
fn shapes(w: &[u8]) -> [bool; 3] {
    let last_wide = w.iter().rposition(|&m| WIDE2 >> m & 1 == 1);
    let mut ok = [true; 3];
    if let Some(lw) = last_wide {
        for &m in &w[..lw] {
            if m < 18 && m % 3 != 1 {
                ok[[1, 0, 2][(m as usize / 3) % 3]] = false;
            }
        }
    }
    ok
}

fn seq_string(p: &[u8]) -> String {
    let ms: Vec<_> = p.iter().map(|&m| move_of(m as usize)).collect();
    format_moves(&ms, Style::Wca)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let b: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(5);
    let samples: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(100);
    let dmin: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(9);
    let dmax: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(11);
    let threads: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(6);
    let k: u32 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(5);
    let canon = Canon::new();

    // The stored side: states u from solved (u = SOLVED * path), keyed; value = the path.
    let t0 = std::time::Instant::now();
    let mut store: Map = Map::default();
    walk(&Cube::SOLVED, b, N_MOVES, &mut vec![], &mut |x, p| {
        let e = store.entry(canon.key(x)).or_insert(u64::MAX);
        if *e == u64::MAX || len_of(*e, b) > p.len() {
            *e = encode(p);
        }
    });
    eprintln!(
        "stored {} states within {b} H-moves in {:.1} s",
        store.len(),
        t0.elapsed().as_secs_f64()
    );
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), k, threads, true);

    // Self-test on a known shortcut: Rw2 U2 D2 F2 B2 Rw2 (6 moves, 3x3x3 distance 8), seen
    // from both sides of the split.
    {
        let x = apply(&Cube::SOLVED, &[22, 1, 10, 7, 16, 22]);
        let x3 = Cube3::from_reduced(&x).unwrap();
        assert_eq!(Search3::new(&t).solve(&x3, 20).unwrap().len(), 8);
        for depth in 0..=1 {
            let mut hit = usize::MAX;
            walk(&x, depth, N_MOVES, &mut vec![], &mut |y, p| {
                if let Some(&e) = store.get(&canon.key(y)) {
                    hit = hit.min(p.len() + len_of(e, b));
                }
            });
            assert!(hit <= 6, "self-test: shortcut not found ({hit})");
        }
    }

    // Test cubes: random canonical outer sequences of length d whose optimal length is d.
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
    let mut tests: Vec<(usize, Cube, Cube3)> = vec![];
    let mut srch = Search3::new(&t);
    for d in dmin..=dmax {
        let mut n = 0;
        while n < samples {
            let mut seq = vec![];
            let mut last = N_MOVES;
            while seq.len() < d {
                let m = rng.gen_range(0..18);
                if ALLOWED_AFTER[last] >> m & 1 == 1 {
                    seq.push(m as u8);
                    last = m;
                }
            }
            let x = apply(&Cube::SOLVED, &seq);
            let x3 = Cube3::from_reduced(&x).unwrap();
            if srch.solve(&x3, d as u8).map(|s| s.len()) == Some(d) {
                tests.push((d, x, x3));
                n += 1;
            }
        }
    }
    eprintln!("{} test cubes", tests.len());

    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![]);
    let t1 = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= tests.len() {
                        break;
                    }
                    let (d, x, _) = &tests[i];
                    if d - 1 < b {
                        continue;
                    }
                    let depth = d - 1 - b;
                    // Shortest solution found, and per axis the shortest of that shape.
                    let mut best: Option<Vec<u8>> = None;
                    let mut best_shape = [usize::MAX; 3];
                    walk(x, depth, N_MOVES, &mut vec![], &mut |y, p| {
                        if let Some(&e) = store.get(&canon.key(y)) {
                            let u = decode(e, len_of(e, b));
                            let mut w = p.to_vec();
                            w.extend(u.iter().rev().map(|&m| inverse_move(m as usize) as u8));
                            if !apply(x, &w).to_facelets().is_solved() {
                                return; // hash collision
                            }
                            let sh = shapes(&w);
                            for a in 0..3 {
                                if sh[a] {
                                    best_shape[a] = best_shape[a].min(w.len());
                                }
                            }
                            if best.as_ref().is_none_or(|bw| w.len() < bw.len()) {
                                best = Some(w);
                            }
                        }
                    });
                    results.lock().unwrap().push((i, best, best_shape));
                }
            });
        }
    });
    eprintln!("searched in {:.1} s", t1.elapsed().as_secs_f64());
    let mut res = results.into_inner().unwrap();
    res.sort_by_key(|r| r.0);
    let mut per_d: BTreeMap<usize, (usize, BTreeMap<usize, usize>, usize, usize)> = BTreeMap::new();
    for (i, best, sh) in &res {
        let d = tests[*i].0;
        let e = per_d.entry(d).or_default();
        e.0 += 1;
        if let Some(w) = best {
            *e.1.entry(d - w.len()).or_default() += 1;
            // Shortcut not found in frame x's shape (by the solutions seen).
            if sh[0] > w.len() {
                e.2 += 1;
            }
            if sh.iter().all(|&l| l > w.len()) {
                e.3 += 1;
            }
            println!(
                "d {d}: H {} [{}] shapes x/y/z {:?}",
                w.len(),
                seq_string(w),
                sh.map(|l| if l == usize::MAX { 0 } else { l })
            );
        }
    }
    println!(
        "d | tested | shortcuts (gain: count) | of which not of frame x's shape | of no frame's shape"
    );
    for (d, (n, g, nx, nall)) in per_d {
        println!("{d} | {n} | {g:?} | {nx} | {nall}");
    }
}
