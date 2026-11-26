//! Random symmetric positions as facelet strings (one per line), for the symmetry benchmarks.
//!   cargo run --release --example symgen -- KIND COUNT SEED
//! KIND: c3 (turns by 120 degrees about the URF-DBL diagonal), m (the reflection swapping U and
//! R), c3v (both), d3 (turns about the UFL-DBR diagonal: pruned after a whole-cube rotation),
//! c4 (turns by 90 degrees about U: not prunable), oh (all 48).
use eggg_solver::cube::{ROTATIONS, mat_apply};
use eggg_solver::symmetry::{PositionSymmetry, SYMS, group, random_symmetric, sym_index};
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map_or("c3", |s| s.as_str());
    let count: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    let seed: u64 = args.get(3).map_or(1, |s| s.parse().unwrap());
    let rot = |from: [i8; 3], to: [i8; 3], fix: [i8; 3]| {
        sym_index(
            ROTATIONS
                .iter()
                .find(|m| mat_apply(m, from) == to && mat_apply(m, fix) == fix)
                .unwrap(),
        )
    };
    let c3 = rot([0, 1, 0], [1, 0, 0], [1, 1, 1]);
    let m = sym_index(&[[0, 1, 0], [1, 0, 0], [0, 0, 1]]);
    let d3 = rot([0, 1, 0], [-1, 0, 0], [-1, 1, 1]);
    let c4 = rot([1, 0, 0], [0, 0, -1], [0, 1, 0]);
    let gens: Vec<usize> = match kind {
        "c3" => vec![c3],
        "m" => vec![m],
        "c3v" => vec![c3, m],
        "d3" => vec![d3],
        "c4" => vec![c4],
        "oh" => (0..SYMS.len()).collect(),
        _ => panic!("unknown kind {kind}"),
    };
    let grp = group(&gens);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    for _ in 0..count {
        let f = random_symmetric(&grp, &mut rng);
        let sym = PositionSymmetry::of(&f.to_cube().unwrap().0, true);
        println!("{}", f.to_string_letters());
        eprintln!("{}", sym.describe());
    }
}
