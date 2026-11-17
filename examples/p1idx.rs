use eggg_solver::cube::Cube;
use eggg_solver::moves::MOVE_PERMS;
use eggg_solver::phase1::P1State;
use eggg_solver::solver::load_tables;
use rand::{Rng, SeedableRng};
fn main() {
    let t = load_tables(std::path::Path::new("tables"), 2, 8, false);
    let st = &t.p1.staging;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
    let mut bad = 0;
    for _ in 0..2000 {
        let mut c = Cube::SOLVED;
        for _ in 0..rng.gen_range(0..12) {
            c = c.apply_perm(&MOVE_PERMS[rng.gen_range(0..27)]);
        }
        let s = P1State::from_cube(&c);
        let (ud, rl) = s.masks();
        let a = st.index(ud, rl);
        let b = st.fast_index(unsafe { *st.info(ud) }, rl);
        let (va, vb) = (t.p1.table.get(a), t.p1.table.get(b));
        if va != vb {
            bad += 1;
            if bad < 5 {
                println!("idx {a} {b}: {va} vs {vb}");
            }
        }
    }
    println!("mismatches {bad}");
}
