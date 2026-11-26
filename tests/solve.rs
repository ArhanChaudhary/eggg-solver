//! End-to-end tests of the solver. They use the tables in $EGGG_TABLES (default ./tables),
//! generating any that are missing (slow the first time), with the H48 table of the small
//! tier (h3; $EGGG_TEST_H48 to change it).

use eggg_solver::cube::{Cube, Facelets};
use eggg_solver::notation::{Style, format_moves, parse_moves};
use eggg_solver::solver::{Options, Solver, Tables, load_tables, verify};
use rand::SeedableRng;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let dir = std::env::var_os("EGGG_TABLES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "tables".into());
        let k = std::env::var("EGGG_TEST_H48")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        load_tables(&dir, k, eggg_solver::par::threads().min(8), false)
    })
}

fn facelets_of(scramble: &str) -> Facelets {
    parse_moves(scramble)
        .unwrap()
        .iter()
        .fold(Cube::SOLVED.to_facelets(), |f, &t| f.apply_turn(t))
}

fn solve(f: &Facelets, secs: f64) -> Vec<u8> {
    let (cube, _) = f.to_cube().unwrap();
    let opts = Options {
        time_limit: Duration::from_secs_f64(secs),
        threads: 4,
        ..Default::default()
    };
    let mut s = Solver::new(tables(), opts);
    let sol = s.solve(&cube, &mut |_| {}).expect("no solution found");
    assert!(verify(f, &sol.moves), "solution does not solve the cube");
    sol.moves
}

#[test]
fn solved_cube_needs_no_moves() {
    assert!(solve(&Cube::SOLVED.to_facelets(), 1.0).is_empty());
    // Rotated solved cubes are solved too.
    assert!(solve(&facelets_of("x y2 z'"), 1.0).is_empty());
}

#[test]
fn short_scrambles_are_solved() {
    for s in [
        "R",
        "Uw",
        "R U",
        "Rw2 F'",
        "3Rw U",
        "r U' 2R",
        "R U R' U'",
        "Fw R2 Dw' L B2",
    ] {
        let sol = solve(&facelets_of(s), 3.0);
        assert!(sol.len() <= 30, "{s}: {} moves", sol.len());
    }
}

#[test]
fn random_states_are_solved() {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
    for _ in 0..2 {
        let f = Cube::random(&mut rng).to_facelets();
        let sol = solve(&f, 2.0);
        assert!(sol.len() <= 60);
    }
}

#[test]
fn facelet_input_and_output_notations() {
    let scramble = "Rw U2 x Rw2 F' Bw D Lw' 2R B2 u'";
    let f = facelets_of(scramble);
    let s = f.to_string_letters();
    let g = Facelets::parse(&s).unwrap();
    assert_eq!(f, g);
    let sol = solve(&g, 3.0);
    let moves: Vec<_> = sol
        .iter()
        .map(|&m| eggg_solver::moves::move_of(m as usize))
        .collect();
    for style in [Style::Wca, Style::Sign] {
        // Printing and parsing back gives a sequence that still solves the cube.
        let text = format_moves(&moves, style);
        let turns = parse_moves(&text).unwrap();
        let end = turns.iter().fold(f, |f, &t| f.apply_turn(t));
        assert!(end.is_solved(), "{style:?}: {text}");
    }
}

/// With `ties`, every reported solution solves the cube, none is longer than the best before
/// it, the same move sequence is never reported twice, and the best solution returned is the
/// first one of the shortest length (the improvements are those without ties). (Whether ties
/// show up within a short run depends on phase 3 reaching the best length in a round.)
#[test]
fn ties_reports_equal_solutions_once() {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
    for _ in 0..2 {
        let f = Cube::random(&mut rng).to_facelets();
        let (cube, _) = f.to_cube().unwrap();
        let opts = Options {
            time_limit: Duration::from_secs_f64(3.0),
            threads: 4,
            ties: true,
            ..Default::default()
        };
        let mut s = Solver::new(tables(), opts);
        let mut seen: Vec<Vec<u8>> = vec![];
        let best = s
            .solve(&cube, &mut |sol| seen.push(sol.moves.clone()))
            .expect("no solution found");
        assert!(!seen.is_empty());
        let mut shortest = usize::MAX;
        for (i, m) in seen.iter().enumerate() {
            assert!(verify(&f, m), "a reported solution does not solve the cube");
            assert!(
                m.len() <= shortest,
                "a reported solution is longer than the best"
            );
            shortest = shortest.min(m.len());
            assert!(!seen[..i].contains(m), "a solution was reported twice");
        }
        assert_eq!(best.len(), shortest);
        let first = seen.iter().find(|m| m.len() == shortest).unwrap();
        assert_eq!(&best.moves, first);
    }
}

/// Symmetric positions (short scrambles with symmetries that fix the URF corner, some only
/// after a whole-cube rotation, and the checkerboard with all 48) are solved as short with the
/// symmetry pruning as without it, the pruning is used, and with `ties` every reported
/// solution (the images of each solution found included) solves the cube, once each.
#[test]
fn symmetric_positions_solve_as_short_with_pruning() {
    for (scramble, used) in [
        ("Uw2", 2),
        ("B2 Fw2", 2),
        ("Fw2 Rw2 Fw2 U2", 2),
        ("Fw2 Rw2 Uw2 Fw2", 3),
        ("R2 L2 U2 D2 F2 B2", 6),
        ("Fw2 Rw2 Uw2 Fw2 R2 L2 U2 D2 F2 B2", 3),
    ] {
        let f = facelets_of(scramble);
        let (cube, _) = f.to_cube().unwrap();
        let mut lens = vec![];
        for symmetry in [true, false] {
            let opts = Options {
                time_limit: Duration::from_secs_f64(1.5),
                threads: 4,
                symmetry,
                ties: true,
                ..Default::default()
            };
            let mut s = Solver::new(tables(), opts);
            let mut seen: Vec<Vec<u8>> = vec![];
            let best = s
                .solve(&cube, &mut |sol| seen.push(sol.moves.clone()))
                .expect("no solution found");
            assert!(verify(&f, &best.moves), "{scramble}");
            for (i, m) in seen.iter().enumerate() {
                assert!(
                    verify(&f, m),
                    "{scramble}: a reported solution does not solve it"
                );
                assert!(!seen[..i].contains(m), "{scramble}: reported twice");
            }
            assert_eq!(
                s.stats.sym_used,
                if symmetry { used } else { 1 },
                "{scramble}"
            );
            if symmetry {
                assert!(s.stats.jobs_pruned > 0, "{scramble}");
            }
            lens.push(best.len());
        }
        assert_eq!(lens[0], lens[1], "{scramble}");
    }
}

/// `Options::stop`, raised from another thread, ends a solve without a time limit within a
/// fraction of a second, mid-run (with the best solution reported so far) and before any
/// solution exists (with none).
#[test]
fn external_stop_ends_a_solve_promptly() {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(99);
    let f = Cube::random(&mut rng).to_facelets();
    let (cube, _) = f.to_cube().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let opts = Options {
        time_limit: Duration::from_secs(3600),
        threads: 4,
        stop: Some(stop.clone()),
        ..Default::default()
    };
    // The length of the last solution reported (0: none yet).
    let last = AtomicUsize::new(0);
    let stopped_at = OnceLock::new();
    let best = std::thread::scope(|s| {
        s.spawn(|| {
            let t = Instant::now();
            while last.load(Ordering::Relaxed) == 0 && t.elapsed() < Duration::from_secs(300) {
                std::thread::sleep(Duration::from_millis(5));
            }
            // Let it search on a little (into the next phases and rounds).
            std::thread::sleep(Duration::from_millis(300));
            stopped_at.set(Instant::now()).unwrap();
            stop.store(true, Ordering::Relaxed);
        });
        Solver::new(tables(), opts.clone()).solve(&cube, &mut |s| {
            last.store(s.len(), Ordering::Relaxed);
        })
    });
    let latency = stopped_at.get().unwrap().elapsed();
    assert!(
        latency < Duration::from_secs(1),
        "stopped after {latency:?}"
    );
    let best = best.expect("no solution returned");
    assert!(
        verify(&f, &best.moves),
        "the solution does not solve the cube"
    );
    assert_eq!(best.len(), last.load(Ordering::Relaxed));
    // Raised before the solve starts: it returns at once, without a solution.
    let t = Instant::now();
    let none = Solver::new(tables(), opts).solve(&cube, &mut |_| {});
    assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
    assert!(none.is_none());
}
