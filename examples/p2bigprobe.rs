//! Time the large phase-2 tables' per-entry generator work (decode and 17 canonical
//! neighbours), without the table data (thread CPU time: the machine may be shared).
use eggg_solver::p2big::{Big, Kind};
use eggg_solver::solver::load_tables_with;
use rand::{Rng, SeedableRng};

fn cpu_now() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

fn main() {
    let t = load_tables_with(std::path::Path::new("tables"), 0, false, false, 4, false);
    for kind in [Kind::UdRl, Kind::UdFb] {
        let t0 = cpu_now();
        let b = Big::new(&t.p2, kind);
        println!(
            "{kind:?}: setup {:.2} s, {} classes, {:.3e} entries, {:.2} GB",
            cpu_now() - t0,
            b.classes(),
            b.n as f64,
            b.bytes() as f64 / 1e9
        );
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
        let idx: Vec<usize> = (0..200_000).map(|_| rng.gen_range(0..b.n)).collect();
        let t0 = cpu_now();
        let mut x = 0;
        for &i in &idx {
            x ^= b.probe(&t.p2, i);
        }
        let per = (cpu_now() - t0) / idx.len() as f64;
        println!(
            "  {:.0} ns CPU per entry expansion ({x:x}); 2 expansions per entry on 48 threads: \
             {:.0} min of compute",
            per * 1e9,
            per * 2.0 * b.n as f64 / 48.0 / 60.0
        );
    }
}
