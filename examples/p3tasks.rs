//! The task-split phase-3 search must agree with the whole search.
use eggg_solver::cube3::Cube3;
use eggg_solver::phase3::{P3Tables, Search3};
use rand::{Rng, SeedableRng};

fn main() {
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), 6, 8, false);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(9);
    let (mut agree, mut n_whole, mut n_split) = (0, 0u64, 0u64);
    for i in 0..40 {
        let len = if i < 20 { rng.gen_range(10..15) } else { 60 };
        let mut x = Cube3::SOLVED;
        for _ in 0..len {
            x = x.apply(rng.gen_range(0..18));
        }
        for d in 12u8..=15 {
            let mut a = Search3::new(&t);
            let whole = a.search_depth(&x, d);
            n_whole += a.nodes;
            let mut b = Search3::new(&t);
            let mut split = None;
            if let Some(tasks) = b.tasks(&x, d) {
                for task in &tasks {
                    if let Some(s) = b.run_task(task) {
                        split = Some(s);
                        break;
                    }
                }
            }
            n_split += b.nodes;
            if let Some(s) = &split {
                assert_eq!(x.apply_moves(s), Cube3::SOLVED);
                assert_eq!(s.len(), d as usize);
            }
            assert_eq!(whole.is_some(), split.is_some(), "cube {i} depth {d}");
            agree += 1;
        }
    }
    println!("{agree} searches agree; nodes whole {n_whole}, split {n_split}");
}
