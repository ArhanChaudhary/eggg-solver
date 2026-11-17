//! Experiment: phase-1 endpoints from "slices" (an A-path that enters staging with a wide
//! quarter turn, then b moves of H = outer turns and wide half turns, which keep the centres
//! staged and need no table lookups) versus the plain level search. Single-threaded rates.
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, N_MOVES, WIDE_QUARTER};
use eggg_solver::phase1::{P1State, P1Tables, Search1, trailing_run};
use eggg_solver::solver::{Tables, endpoint_h2, load_tables_with};
use eggg_solver::table::Mod3Table;
use rand::SeedableRng;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

type Hist = BTreeMap<(usize, usize), usize>;

fn record(t: &Tables, cube: &Cube, path: &[u8], frames: u8, hist: &mut Hist, n: &mut usize) {
    for f in 0..3 {
        if frames >> f & 1 == 1 {
            *n += 1;
            if let Some(h2) = endpoint_h2(t, cube, path, f) {
                *hist
                    .entry((path.len(), path.len() + h2 as usize))
                    .or_default() += 1;
            }
        }
    }
}

/// A-leaves: staged, even parity, last move a wide quarter turn, exactly `left` more moves.
#[allow(clippy::too_many_arguments)]
fn aleaves(
    p1: &P1Tables,
    s: &P1State,
    d: u8,
    left: u8,
    path: &mut Vec<u8>,
    deadline: Instant,
    nodes: &mut u64,
    f: &mut dyn FnMut(&P1State, &mut Vec<u8>),
) -> bool {
    if left == 0 {
        let last = path.last().map(|&m| m as usize);
        if d == 0 && !s.parity && last.is_none_or(|m| WIDE_QUARTER >> m & 1 == 1) {
            f(s, path);
        }
        return true;
    }
    if *nodes & 4095 == 0 && Instant::now() > deadline {
        return false;
    }
    let last = path.last().map(|&m| m as usize).unwrap_or(N_MOVES);
    let mut allowed = ALLOWED_AFTER[last];
    if left == 1 {
        allowed &= WIDE_QUARTER;
    }
    while allowed != 0 {
        let m = allowed.trailing_zeros() as usize;
        allowed &= allowed - 1;
        *nodes += 1;
        let c = s.apply(m, p1);
        let dc = Mod3Table::decode(d, p1.value(&c));
        if dc < left {
            path.push(m as u8);
            let go = aleaves(p1, &c, dc, left - 1, path, deadline, nodes, f);
            path.pop();
            if !go {
                return false;
            }
        }
    }
    true
}

/// H-paths of exactly `left` moves; endpoints by the frame rule (no table lookups).
fn hext(
    p1: &P1Tables,
    s: &P1State,
    left: u8,
    path: &mut Vec<u8>,
    tests: &mut u64,
    f: &mut dyn FnMut(&[u8], u8),
) {
    if left == 0 {
        *tests += 1;
        let fr = p1.endpoint_frames(s, trailing_run(path));
        if fr != 0 {
            f(path, fr);
        }
        return;
    }
    let last = path.last().map(|&m| m as usize).unwrap_or(N_MOVES);
    let mut allowed = ALLOWED_AFTER[last] & !WIDE_QUARTER;
    while allowed != 0 {
        let m = allowed.trailing_zeros() as usize;
        allowed &= allowed - 1;
        let c = s.apply(m, p1);
        path.push(m as u8);
        hext(p1, &c, left - 1, path, tests, f);
        path.pop();
    }
}

fn summary(name: &str, secs: f64, n: usize, hist: &Hist, l12_max: usize) {
    let useful: usize = hist
        .iter()
        .filter(|(k, _)| k.1 <= l12_max)
        .map(|(_, v)| v)
        .sum();
    let byl: BTreeMap<usize, usize> =
        hist.iter()
            .filter(|(k, _)| k.1 <= l12_max)
            .fold(BTreeMap::new(), |mut m, (k, v)| {
                *m.entry(k.1).or_default() += v;
                m
            });
    println!(
        "{name:28} {secs:7.2} s  endpoints {n:8} ({:9.0}/s)  L1+h2<={l12_max}: {useful:6} ({:8.1}/s) {byl:?}",
        n as f64 / secs,
        useful as f64 / secs
    );
}

fn main() {
    let budget: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20.0);
    let t = load_tables_with(std::path::Path::new("tables"), 3, true, true, 8, false);
    let p1 = &t.p1;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(77);
    for _ in 0..2 {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        // The first level with endpoints, then the next two (the ones the solver works on).
        let mut first = d0 as usize;
        loop {
            let mut n = 0;
            let mut srch = Search1::new(p1, |_: &[u8], _| {
                n += 1;
                true
            });
            srch.dfs(&root, d0, first as u8, N_MOVES);
            if n > 0 {
                break;
            }
            first += 1;
        }
        println!("--- cube: staging d0 {d0}, first endpoints at L1 = {first}");
        for deep in [first + 1, first + 2] {
            let l12_max = deep + 9;
            {
                let stop = AtomicBool::new(false);
                let done = AtomicBool::new(false);
                let (mut hist, mut n) = (Hist::new(), 0usize);
                let mut eps = vec![];
                let t0 = Instant::now();
                std::thread::scope(|sc| {
                    let (stop, done) = (&stop, &done);
                    sc.spawn(move || {
                        while !done.load(Ordering::Relaxed) {
                            if t0.elapsed().as_secs_f64() > budget {
                                stop.store(true, Ordering::Relaxed);
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(5));
                        }
                    });
                    let mut srch = Search1::new(p1, |p: &[u8], f| {
                        eps.push((p.to_vec(), f));
                        true
                    });
                    srch.abort = Some(stop);
                    srch.dfs(&root, d0, deep as u8, N_MOVES);
                    done.store(true, Ordering::Relaxed);
                });
                let secs = t0.elapsed().as_secs_f64();
                for (p, f) in &eps {
                    record(&t, &cube, p, *f, &mut hist, &mut n);
                }
                summary(&format!("plain L1={deep}"), secs, n, &hist, l12_max);
            }
            // Slices (a, b) with a in {d0, d0+1}.
            let fa = first as u8;
            for (a, b) in [
                (fa - 1, (deep as u8) - fa + 1),
                (fa, (deep as u8) - fa),
                (fa + 1, (deep as u8) - fa - 1),
            ] {
                if b == 0 {
                    continue;
                }
                let (mut hist, mut n, mut tests, mut nodes, mut leaves) =
                    (Hist::new(), 0usize, 0u64, 0u64, 0u64);
                let mut eps = vec![];
                let t0 = Instant::now();
                let deadline = t0 + Duration::from_secs_f64(budget);
                let done = aleaves(
                    p1,
                    &root,
                    d0,
                    a,
                    &mut vec![],
                    deadline,
                    &mut nodes,
                    &mut |s, path| {
                        leaves += 1;
                        hext(p1, s, b, path, &mut tests, &mut |p, f| {
                            eps.push((p.to_vec(), f))
                        });
                    },
                );
                let secs = t0.elapsed().as_secs_f64();
                for (p, f) in &eps {
                    record(&t, &cube, p, *f, &mut hist, &mut n);
                }
                summary(
                    &format!("slice a={a} b={b}{}", if done { "" } else { " (cut)" }),
                    secs,
                    n,
                    &hist,
                    l12_max,
                );
                println!(
                    "{:28} A-leaves {leaves}, A nodes {nodes}, leaf tests {tests}",
                    ""
                );
            }
        }
    }
}
