use eggg_solver::cube::Cube;
use eggg_solver::phase1::P1State;
use eggg_solver::solver::load_tables;
use eggg_solver::table::{HugeBuf, Mod3Table};
use rand::SeedableRng;
fn main() {
    let t = load_tables(std::path::Path::new("tables"), 2, 8, false);
    let st = &t.p1.staging;
    let n = eggg_solver::phase1::P1Tables::table_len();
    let mut old = Mod3Table {
        data: HugeBuf::zeroed(n.div_ceil(32)),
        n: n as u64,
    };
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            old.data.as_mut_slice().as_mut_ptr() as *mut u8,
            old.data.byte_len(),
        )
    };
    eggg_solver::table::load(
        std::path::Path::new("tables/p1-staging.tab"),
        "p1-staging",
        bytes,
    )
    .unwrap();
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
    let (mut bad, mut bad2) = (0, 0);
    for _ in 0..20000 {
        let c = Cube::random(&mut rng);
        let s = P1State::from_cube(&c);
        let (ud, rl) = s.masks();
        let a = st.index(ud, rl);
        let b = st.fast_index(unsafe { *st.info(ud) }, rl);
        if old.get(a) != t.p1.table.get(a) {
            bad += 1;
        }
        if t.p1.table.get(a) != t.p1.table.get(b) {
            bad2 += 1;
            if bad2 < 4 {
                let (u2, r2) = st.rep_of(b);
                println!(
                    "a {a} b {b} va {} vb {} index(rep_of(b)) {} class {} split a {} b {}",
                    t.p1.table.get(a),
                    t.p1.table.get(b),
                    st.index(u2, r2),
                    a / 6435,
                    a % 6435,
                    b % 6435
                );
            }
        }
    }
    println!("old vs new (canonical): {bad}; new canonical vs fast: {bad2}");
    let mut hist_old = [0u64; 4];
    let mut hist_new = [0u64; 4];
    for i in (0..n as u64).step_by(997) {
        hist_old[old.get(i) as usize] += 1;
        hist_new[t.p1.table.get(i) as usize] += 1;
    }
    println!("code histograms old {:?} new {:?}", hist_old, hist_new);
}
