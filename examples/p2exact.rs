//! The exact joint phase-2 table against the mod-3 one: bounded distances of random phase-2
//! states (both views) by one read of the exact table and by walks down the mod-3 table must
//! agree, and their times per state. Generates the exact table (p2-joint-exact.tab) if it is
//! missing. Usage: p2exact [states] [threads] [seed] (tables in $EGGG_TABLES or ./tables).
use eggg_solver::cube::Cube;
use eggg_solver::moves::MOVE_PERMS;
use eggg_solver::phase2::P2_MOVES;
use eggg_solver::solver::{JointTable, load_p2};
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let arg = |i: usize, d: u64| -> u64 {
        std::env::args()
            .nth(i)
            .map(|s| s.parse().unwrap())
            .unwrap_or(d)
    };
    let (count, threads, seed) = (arg(1, 200_000) as usize, arg(2, 8) as usize, arg(3, 1));
    let dir = std::env::var_os("EGGG_TABLES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "tables".into());
    let t0 = Instant::now();
    let pe = load_p2(&dir, JointTable::Exact, threads, true);
    eprintln!("exact table ready in {:.1} s", t0.elapsed().as_secs_f64());
    let pm = load_p2(&dir, JointTable::Mod3, threads, true);
    let (je, jm) = (pe.joint.as_ref().unwrap(), pm.joint.as_ref().unwrap());
    assert!(je.exact && !jm.exact);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    // Random states (moves from solved: 10 to 40, so most are deep) and both views, with
    // bounds around their distances (as candidate making has them).
    let mut states = Vec::with_capacity(2 * count);
    for _ in 0..count {
        let mut c = Cube::SOLVED;
        for _ in 0..rng.gen_range(10..40) {
            c = c.apply_perm(&MOVE_PERMS[P2_MOVES[rng.gen_range(0..17)] as usize]);
        }
        let cc = pm.centre_coord_of(&c);
        let tau = pm.tau_of(&c);
        let max = rng.gen_range(6..16);
        states.push((cc / 1680, tau, cc & 1, max));
        states.push((jm.fb.ud(cc), jm.fb.tau(&tau), cc & 1, max));
    }
    let (mut walked, mut read) = (vec![], vec![]);
    // Chunks of 768 states (the solver makes candidates 256 endpoints at a time, up to three
    // frames each).
    let time = |j: &eggg_solver::p2joint::Joint, out: &mut Vec<Option<u8>>| {
        let t = Instant::now();
        let mut part = vec![];
        for chunk in states.chunks(768) {
            j.distances_at_most(chunk, &mut part);
            out.extend_from_slice(&part);
        }
        t.elapsed().as_secs_f64()
    };
    // Twice each, the second timed (warm caches and page tables).
    time(jm, &mut walked);
    time(je, &mut read);
    walked.clear();
    read.clear();
    let tw = time(jm, &mut walked);
    let tr = time(je, &mut read);
    let bad = walked.iter().zip(&read).filter(|(a, b)| a != b).count();
    let within = walked.iter().filter(|d| d.is_some()).count();
    let mut hist = [0usize; 16];
    for i in (0..states.len()).step_by(2) {
        let (ud, tau, pll, _) = states[i];
        hist[je.get_exact(je.index(ud, &tau, pll)) as usize] += 1;
    }
    println!(
        "{} states ({within} within their bound): walk {:.1} ns/state, read {:.1} ns/state ({:.1}x); mismatches {bad}",
        states.len(),
        tw * 1e9 / states.len() as f64,
        tr * 1e9 / states.len() as f64,
        tw / tr
    );
    println!("U/D distances of the states: {hist:?}");
    assert_eq!(bad, 0);
}
