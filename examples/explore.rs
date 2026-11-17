use eggg_solver::cube::*;
use std::collections::{HashSet, VecDeque};

fn main() {
    // Phase-2 generators (special axis x): U D F B quarter, R2 L2, Uw2 Rw2 Fw2.
    let mut gens = vec![];
    for f in [U, D, F, B] {
        gens.push(Turn::face(f, 1, 1));
    }
    for f in [R, L] {
        gens.push(Turn::face(f, 1, 2));
    }
    for f in [U, R, F] {
        gens.push(Turn::face(f, 2, 2));
    }
    let perms: Vec<Perm> = gens.iter().map(|&t| Perm::of_turn(t)).collect();
    // Wing position orbits.
    let mut orbit = [usize::MAX; 24];
    let mut norb = 0;
    for s in 0..24 {
        if orbit[s] != usize::MAX {
            continue;
        }
        let mut q = vec![s];
        orbit[s] = norb;
        while let Some(x) = q.pop() {
            for p in &perms {
                // p.wing_src[t] = s means slot s moves to t
                for t in 0..24 {
                    if p.wing_src[t] as usize == x && orbit[t] == usize::MAX {
                        orbit[t] = norb;
                        q.push(t);
                    }
                }
            }
        }
        norb += 1;
    }
    println!("wing orbits: {norb} {:?}", orbit);
    // Centre orbits
    let mut corb = [usize::MAX; 24];
    let mut nc = 0;
    for s in 0..24 {
        if corb[s] != usize::MAX {
            continue;
        }
        let mut q = vec![s];
        corb[s] = nc;
        while let Some(x) = q.pop() {
            for p in &perms {
                for t in 0..24 {
                    if p.centre_src[t] as usize == x && corb[t] == usize::MAX {
                        corb[t] = nc;
                        q.push(t);
                    }
                }
            }
        }
        nc += 1;
    }
    println!("centre orbits: {nc} {:?}", corb);
    // Reachable 2-colour patterns per orbit from the solved pattern, also including K4 rotations
    for o in 0..nc {
        let pos: Vec<usize> = (0..24).filter(|&s| corb[s] == o).collect();
        let start: Vec<u8> = (0..24).map(|s| Cube::SOLVED.ct[s] / 4).collect();
        let key = |c: &[u8]| pos.iter().map(|&s| c[s]).collect::<Vec<u8>>();
        let mut seen = HashSet::new();
        let mut q = VecDeque::new();
        seen.insert(key(&start));
        q.push_back(start.clone());
        while let Some(c) = q.pop_front() {
            for p in &perms {
                let n: Vec<u8> = (0..24).map(|t| c[p.centre_src[t] as usize]).collect();
                if seen.insert(key(&n)) {
                    q.push_back(n);
                }
            }
        }
        println!(
            "orbit {o} positions {:?}: {} reachable patterns",
            pos,
            seen.len()
        );
    }
    // Rotations: x, y2 etc. effect on wing orbits
    for (name, t) in [
        ("x", Turn::rotation(R, 1)),
        ("y", Turn::rotation(U, 1)),
        ("y2", Turn::rotation(U, 2)),
        ("z", Turn::rotation(F, 1)),
    ] {
        let p = Perm::of_turn(t);
        let mut same = 0;
        for t2 in 0..24 {
            let s = p.wing_src[t2] as usize;
            if orbit[s] == orbit[t2] {
                same += 1;
            }
        }
        println!("{name}: wings keeping orbit {same}/24");
    }
}
