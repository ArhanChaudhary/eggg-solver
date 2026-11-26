//! End-to-end tests of `eggg-solver interactive`: commands written to its stdin one at a time,
//! results read from its stdout. The tables are those of tests/solve.rs ($EGGG_TABLES, default
//! ./tables), in the small tier (H48 h3; $EGGG_TEST_H48 to change it).

use eggg_solver::cube::{Cube, Facelets};
use eggg_solver::notation::parse_moves;
use rand::SeedableRng;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

/// A running `eggg-solver interactive`, its output lines as they come: (stderr, line).
struct Repl {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<(bool, String)>,
    /// Every line read so far (for failure messages).
    seen: Vec<String>,
}

impl Repl {
    fn start() -> Repl {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_eggg-solver"));
        cmd.args(["interactive", "--tier", "small", "-j", "4"]);
        if let Ok(k) = std::env::var("EGGG_TEST_H48") {
            cmd.args(["--h48", &k]);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("cannot run eggg-solver");
        let (tx, lines) = channel();
        for err in [false, true] {
            let tx = tx.clone();
            let out: Box<dyn std::io::Read + Send> = if err {
                Box::new(child.stderr.take().unwrap())
            } else {
                Box::new(child.stdout.take().unwrap())
            };
            std::thread::spawn(move || {
                for l in BufReader::new(out).lines().map_while(Result::ok) {
                    if tx.send((err, l)).is_err() {
                        break;
                    }
                }
            });
        }
        Repl {
            stdin: child.stdin.take(),
            child,
            lines,
            seen: vec![],
        }
    }

    fn send(&mut self, cmd: &str) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{cmd}").unwrap();
        stdin.flush().unwrap();
    }

    /// The lines up to the first one (stdout or stderr: `err`) that satisfies `pred`, which
    /// is last.
    fn until(&mut self, err: bool, pred: impl Fn(&str) -> bool) -> Vec<String> {
        let end = Instant::now() + Duration::from_secs(300);
        let mut out = vec![];
        loop {
            let left = end.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok((e, l)) => {
                    self.seen
                        .push(format!("{}{l}", if e { "[stderr] " } else { "" }));
                    let hit = e == err && pred(&l);
                    if !e {
                        out.push(l);
                    }
                    if hit {
                        if e {
                            out.push(self.seen.last().unwrap().clone());
                        }
                        return out;
                    }
                }
                Err(_) => panic!(
                    "expected output not seen; output:\n{}",
                    self.seen.join("\n")
                ),
            }
        }
    }

    /// Close stdin (end of the input) and wait for the process to exit.
    fn finish(mut self) -> Vec<String> {
        drop(self.stdin.take());
        let t = Instant::now();
        let status = self.child.wait().unwrap();
        assert!(status.success(), "exit status {status}");
        assert!(t.elapsed() < Duration::from_secs(10), "slow exit");
        let rest: Vec<String> = self.lines.iter().map(|x| x.1).collect();
        self.seen.extend(rest.iter().cloned());
        rest
    }
}

fn facelets_of(scramble: &str) -> Facelets {
    parse_moves(scramble)
        .unwrap()
        .iter()
        .fold(Cube::SOLVED.to_facelets(), |f, &t| f.apply_turn(t))
}

/// `moves` (as printed) solves `f`.
fn solves(f: &Facelets, moves: &str) -> bool {
    let turns = parse_moves(moves).unwrap();
    turns.iter().fold(*f, |f, &t| f.apply_turn(t)).is_solved()
}

/// An improvement line: `time s  length moves  phases [..]  moves`.
fn is_improvement(l: &str) -> bool {
    l.contains(" s ") && l.contains(" moves  phases [")
}

impl Repl {
    /// The solution printed before the line that ends a solve (`N moves; found at ...`, the
    /// last of `out`), checked against the last improvement printed.
    fn final_solution(&self, out: &[String]) -> String {
        let n = out.len();
        assert!(n >= 2, "{out:?}");
        let sol = out[n - 2].clone();
        let len: usize = out[n - 1].split(' ').next().unwrap().parse().unwrap();
        assert_eq!(parse_moves(&sol).unwrap().len(), len, "{out:?}");
        let last = self.seen.iter().rev().find(|l| is_improvement(l)).unwrap();
        assert!(last.ends_with(&format!("  {sol}")), "{last} / {sol}");
        sol
    }
}

/// `solve SCRAMBLE -t 2`: improvements, then the best solution and why the solve ended (here
/// its time limit; `wait` waits for that); `quit` exits.
#[test]
fn solve_prints_a_verified_solution() {
    let scramble = "Rw U2 x Rw2 F' Bw D Lw' 2R B2 u'";
    let mut r = Repl::start();
    r.send(&format!("solve {scramble} -t 2"));
    r.send("wait");
    r.send("quit");
    let out = r.until(false, |l| l.contains("time limit reached after"));
    assert_eq!(out[0], format!("# {scramble}"));
    assert!(
        out[1..out.len() - 2].iter().all(|l| is_improvement(l)),
        "{out:?}"
    );
    let sol = r.final_solution(&out);
    assert!(solves(&facelets_of(scramble), &sol), "{sol}");
    assert!(r.finish().is_empty());
}

/// `random 1` then `stop` once it has a solution: the best so far is printed. Then invalid
/// commands and positions print an error and the session goes on; a new solve while one runs
/// stops it first; the end of the input stops the running solve and exits.
#[test]
fn stop_errors_and_restarts() {
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
    let f = Cube::random(&mut rng).to_facelets();
    let mut r = Repl::start();
    r.send("random 1");
    let out = r.until(false, is_improvement);
    assert_eq!(
        out[0],
        format!("# random (seed 1): {}", f.to_string_letters())
    );
    r.send("status");
    let out = r.until(false, |l| l.starts_with("running for "));
    assert!(out.last().unwrap().contains(", best "), "{out:?}");
    r.send("stop");
    let out = r.until(false, |l| l.contains("; stopped after "));
    assert!(solves(&f, &r.final_solution(&out)));
    r.send("status");
    r.until(false, |l| l == "no solve running");
    r.send("stop");
    r.until(false, |l| l == "no solve running");

    r.send("bogus 12");
    r.until(true, |l| l.contains("unknown command 'bogus'"));
    r.send("solve R U Q");
    r.until(true, |l| l.starts_with("error: invalid position"));
    r.send("solve R U -t x");
    r.until(true, |l| l.starts_with("error: invalid value 'x'"));
    r.send("help");
    // (Its last line.)
    r.until(false, |l| l.trim_start().starts_with("help "));

    // A new solve stops the running one first; -l ends it at the first short enough solution.
    r.send("random 2");
    r.send("solve R U' F2 -l 30");
    let out = r.until(false, |l| l.contains("stopped after"));
    assert!(out[0].starts_with("# random (seed 2): "), "{out:?}");
    let out = r.until(false, |l| l.contains("-l reached after"));
    assert_eq!(out[0], "# R U' F2");
    assert!(solves(&facelets_of("R U' F2"), &r.final_solution(&out)));

    r.send("random 3");
    r.until(false, |l| l.starts_with("# random (seed 3)"));
    let rest = r.finish();
    assert!(
        rest.last().is_some_and(|l| l.contains("stopped after")),
        "{rest:?}"
    );
}
