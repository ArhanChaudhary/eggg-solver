//! Build the phase-1 wing-path table (see `eggg_solver::wingpath`) and report its size and
//! build time. K, S: the largest distances for tight and slack-1 lists (default 7 and 5);
//! COLLAPSE=1: then back it with huge pages (timed); DIGEST=1: also a digest of every state's lists (with per-distance statistics), for
//! comparing builders.
use eggg_solver::phase1::Frame;
use eggg_solver::wingpath::WingPaths;
use std::time::Instant;

fn main() {
    let env = |k: &str, d: u8| std::env::var(k).ok().map_or(d, |v| v.parse().unwrap());
    let k = env("K", 7);
    let s = env("S", 5).min(k);
    let frames = [0, 1, 2].map(Frame::new);
    let t0 = Instant::now();
    let w = WingPaths::new(&frames, k, s, false);
    println!(
        "K {k} S {s}: {} bytes, built in {:.2} s",
        w.bytes(),
        t0.elapsed().as_secs_f64()
    );
    if std::env::var_os("COLLAPSE").is_some() {
        let t1 = Instant::now();
        w.collapse();
        println!("collapsed in {:.2} s", t1.elapsed().as_secs_f64());
    }
    if std::env::var_os("DIGEST").is_some() {
        let t1 = Instant::now();
        let (digest, stats) = w.digest();
        for (d, [n, t, s]) in stats.iter().enumerate() {
            println!("distance {d}: {n} states, {t} tight pairs, {s} slack-1 pairs");
        }
        println!("digest {digest:016x} ({:.1} s)", t1.elapsed().as_secs_f64());
    }
}
