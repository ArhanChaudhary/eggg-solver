//! Microbenchmark of the phase-1 wing lookahead (`Wing2`): the per-node test as in the
//! search, the same without bounds checks, and the node-parallel (structure-of-arrays) test
//! over groups of up to 32 nodes, on captured inputs (`p1check` built with `--features
//! p1prof`, `EGGG_W2_CAPTURE=N EGGG_W2_FILE=path`). Usage: w2bench FILE [REPS]. Prints the
//! time and (where perf counters are readable) user instructions and cycles per test.
use eggg_solver::phase1::{P1Tables, Staging};
use eggg_solver::simd::V32;
use eggg_solver::table::Mod3Table;
use std::time::Instant;

#[path = "common/perfc.rs"]
mod perf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let reps: usize = args.get(2).map_or(5, |v| v.parse().unwrap());
    let tests: Vec<(V32, bool, bool)> = data
        .chunks_exact(25)
        .map(|c| (V32::from_slice(&c[..24]), c[24] & 1 == 1, c[24] & 2 == 2))
        .collect();
    let n = tests.len();
    let t = P1Tables::with_table(Staging::new(), Mod3Table::from_depths(&[0]));
    let w = &*t.wing2;
    let mut classes = [0usize; 4];
    for &(_, odd, tight) in &tests {
        classes[odd as usize + 2 * !tight as usize] += 1;
    }
    println!("{n} tests; tight even/odd, slack even/odd: {classes:?}");
    let want: Vec<bool> = tests
        .iter()
        .map(|(p, o, ti)| w.possible(p, *o, *ti))
        .collect();
    println!("pass {}", want.iter().filter(|&&x| x).count());
    let (ins, cyc) = (
        perf::Counter::new("instructions"),
        perf::Counter::new("cycles"),
    );
    let run = |name: &str, f: &mut dyn FnMut() -> Vec<bool>| {
        let mut best = (f64::MAX, 0u64, 0u64);
        for _ in 0..reps {
            if let (Some(i), Some(c)) = (&ins, &cyc) {
                i.start();
                c.start();
            }
            let t0 = Instant::now();
            let got = std::hint::black_box(f());
            let dt = t0.elapsed().as_secs_f64();
            let (ni, nc) = match (&ins, &cyc) {
                (Some(i), Some(c)) => (i.stop(), c.stop()),
                _ => (0, 0),
            };
            assert!(got == want, "{name}: results differ");
            if dt < best.0 {
                best = (dt, ni, nc);
            }
        }
        println!(
            "{name:>28}: {:7.2} ns/test, {:7.1} instr/test, {:7.1} cycles/test, IPC {:.2}",
            best.0 * 1e9 / n as f64,
            best.1 as f64 / n as f64,
            best.2 as f64 / n as f64,
            best.1 as f64 / best.2.max(1) as f64
        );
    };
    run("per node (search)", &mut || {
        tests
            .iter()
            .map(|(p, o, ti)| w.possible(p, *o, *ti))
            .collect()
    });
    run("per node, unchecked", &mut || {
        tests
            .iter()
            .map(|(p, o, ti)| w.possible_unchecked(p, *o, *ti))
            .collect()
    });
    run("per node, packed tight-even", &mut || {
        tests
            .iter()
            .map(|(p, o, ti)| w.possible_packed(p, *o, *ti))
            .collect()
    });
    // Node-parallel: the tests in order, queued by class (parity, and tight or not, or
    // parity only); a full queue (32) is transposed and tested.
    for by_tight in [true, false] {
        let name = if by_tight {
            "SoA x32, by parity+tight"
        } else {
            "SoA x32, by parity"
        };
        run(name, &mut || {
            let mut out = vec![false; n];
            let mut q: [(usize, [V32; 32], u32); 4] = [(0, [V32::ZERO; 32], 0); 4];
            let mut qi = [[0u32; 32]; 4];
            let flush =
                |q: &mut (usize, [V32; 32], u32), qi: &[u32; 32], odd: bool, out: &mut [bool]| {
                    let k = q.0;
                    if k == 0 {
                        return;
                    }
                    let mut m = q.1;
                    eggg_solver::simd::transpose32(&mut m);
                    let lanes = if k == 32 { !0 } else { (1u32 << k) - 1 };
                    let r = w.possible_soa(&m[..24], odd, q.2, lanes);
                    for j in 0..k {
                        out[qi[j] as usize] = r >> j & 1 == 1;
                    }
                    q.0 = 0;
                    q.2 = 0;
                };
            for (i, &(p, odd, tight)) in tests.iter().enumerate() {
                let c = odd as usize + if by_tight { 2 * !tight as usize } else { 0 };
                let qq = &mut q[c];
                qq.1[qq.0] = p;
                qq.2 |= (tight as u32) << qq.0;
                qi[c][qq.0] = i as u32;
                qq.0 += 1;
                if qq.0 == 32 {
                    flush(qq, &qi[c], odd, &mut out);
                }
            }
            for c in 0..4 {
                flush(&mut q[c], &qi[c], c & 1 == 1, &mut out);
            }
            out
        });
    }
    // The kernel alone on pre-transposed full groups of one class (upper bound of the gain).
    for (cls, name) in [
        (0usize, "tight even"),
        (1, "tight odd"),
        (2, "slack even"),
        (3, "slack odd"),
    ] {
        let sel: Vec<&(V32, bool, bool)> = tests
            .iter()
            .filter(|(_, o, ti)| *o as usize + 2 * !*ti as usize == cls)
            .collect();
        let groups: Vec<[V32; 32]> = sel
            .chunks_exact(32)
            .map(|g| {
                let mut m: [V32; 32] = std::array::from_fn(|j| g[j].0);
                eggg_solver::simd::transpose32(&mut m);
                m
            })
            .collect();
        if groups.is_empty() {
            continue;
        }
        let (odd, tight) = (cls & 1 == 1, cls < 2);
        let tm = if tight { !0 } else { 0 };
        let k = groups.len() * 32;
        let i0 = Instant::now();
        let ic = ins.as_ref().map(|c| {
            c.start();
            c
        });
        let mut acc = 0u32;
        for _ in 0..reps {
            for g in &groups {
                acc = acc.wrapping_add(
                    w.possible_soa(std::hint::black_box(&g[..24]), odd, tm, !0)
                        .count_ones(),
                );
            }
        }
        let ni = ic.map_or(0, |c| c.stop());
        let dt = i0.elapsed().as_secs_f64() / reps as f64;
        let i1 = Instant::now();
        let ic = ins.as_ref().map(|c| {
            c.start();
            c
        });
        let mut acc2 = 0u32;
        for _ in 0..reps {
            for g in sel.chunks_exact(32) {
                for x in g {
                    acc2 += w.possible(std::hint::black_box(&x.0), odd, tight) as u32;
                }
            }
        }
        let ni2 = ic.map_or(0, |c| c.stop());
        let dt2 = i1.elapsed().as_secs_f64() / reps as f64;
        assert_eq!(acc, acc2);
        println!(
            "  kernel {name:>10}: SoA {:6.2} ns {:6.1} instr | per node {:6.2} ns {:6.1} instr (per test, {k} tests)",
            dt * 1e9 / k as f64,
            ni as f64 / (reps * k) as f64,
            dt2 * 1e9 / k as f64,
            ni2 as f64 / (reps * k) as f64
        );
    }
}
