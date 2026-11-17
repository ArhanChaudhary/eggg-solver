use eggg_solver::cube::Cube;
use eggg_solver::moves::MOVE_PERMS;
use eggg_solver::phase2::P2_MOVES;
use eggg_solver::solver::load_tables_with;
use std::collections::HashMap;

fn main() {
    let t = load_tables_with(std::path::Path::new("tables"), 2, true, 8, false);
    let p2 = &t.p2;
    let j = p2.joint.as_ref().unwrap();
    // Raw BFS from solved over full cubes restricted to the coordinate (ud, tau, pll).
    let key = |c: &Cube| {
        let cc = p2.centre_coord_of(c);
        (cc / 1680, p2.tau_of(c), cc & 1)
    };
    let mut depth: HashMap<(u32, [u8; 16], u32), u8> = HashMap::new();
    let mut frontier = vec![Cube::SOLVED];
    // both goal orientations: U/D swapped centres (x2 recolour) is a goal too; start from the
    // solved cube and the solved cube with U/D centre colours swapped.
    let mut swapped = Cube::SOLVED;
    for s in [0usize, 1, 2, 3, 12, 13, 14, 15] {
        swapped.ct[s] = if s < 4 { 12 + s as u8 } else { s as u8 - 12 };
    }
    frontier.push(swapped);
    for c in &frontier {
        depth.insert(key(c), 0);
    }
    for d in 1..=6u8 {
        let mut next = vec![];
        for c in &frontier {
            for &m in &P2_MOVES {
                let n = c.apply_perm(&MOVE_PERMS[m as usize]);
                let k = key(&n);
                if !depth.contains_key(&k) {
                    depth.insert(k, d);
                    next.push(n);
                }
            }
        }
        frontier = next;
        eprintln!("raw depth {d}: {}", frontier.len());
    }
    let mut bad = 0;
    let mut checked = 0;
    for (k, &d) in depth.iter() {
        checked += 1;
        let jd = j.distance(p2, k.0, &k.1, k.2);
        if jd != d {
            bad += 1;
            if bad <= 8 {
                let i = j.index(k.0, &k.1, k.2);
                let c = j.canonical_pub(k.0, &k.1, k.2);
                println!(
                    "ud {} pll {} raw {d} joint {jd} index {i} code {} canonical {c} code {}",
                    k.0,
                    k.2,
                    j.get(i),
                    j.get(c)
                );
            }
        }
    }
    println!("checked {checked}, mismatches {bad}");
    let mut by_depth = [0usize; 8];
    let mut tot = [0usize; 8];
    for (k, &d) in depth.iter() {
        tot[d as usize] += 1;
        let i = j.index(k.0, &k.1, k.2);
        if j.get(i) != d % 3 {
            by_depth[d as usize] += 1;
        }
    }
    println!("wrong codes by raw depth: {:?} of {:?}", by_depth, tot);
}
