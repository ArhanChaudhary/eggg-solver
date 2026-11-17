use eggg_solver::cube3::Cube3;
use eggg_solver::phase3::{P3Tables, Search3};
use rand::{Rng, SeedableRng};

fn main() {
    let k: u32 = std::env::args()
        .nth(1)
        .map(|s| s.parse().unwrap())
        .unwrap_or(2);
    let t = P3Tables::load_or_build(std::path::Path::new("tables"), k, 8, true);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
    let mut bad = 0;
    for len in 1..=12usize {
        for _ in 0..3000 {
            let mut x = Cube3::SOLVED;
            let mut last = 99;
            let mut n = 0;
            while n < len {
                let m = rng.gen_range(0..18);
                if last != 99 && m / 3 == last / 3 {
                    continue;
                }
                x = x.apply(m);
                last = m;
                n += 1;
            }
            let s = Search3::new(&t);
            let b = s.lower_bound(&x);
            if b as usize > len {
                bad += 1;
                if bad < 10 {
                    println!(
                        "len {len}: bound {b} (h48 n {} i {})",
                        t.bound_pub(&x),
                        t.bound_pub(&x.inverse())
                    );
                }
            }
        }
    }
    println!("inadmissible bounds: {bad}");
}
