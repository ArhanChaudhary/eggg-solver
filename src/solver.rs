//! The three-phase driver.
//!
//! 1. Phase 1 collects endpoints by increasing length (one search tree serves all 18
//!    frame/colour-assignment combinations); each length level is split into jobs and can be
//!    resumed, so no endpoint is searched twice.
//! 2. Each endpoint is brought into the standard frame, and phase 2 is searched level by level
//!    on the reduction length, collecting reductions (deduplicated by their 3x3x3).
//! 3. Phase 3 first finds a solution quickly (a bounded IDA* on the most promising
//!    reductions), then searches level by level on the total length over all reductions, so
//!    each solution found is the shortest the pool allows.
//!
//! Rounds repeat with geometrically more endpoints and reductions while time remains, only
//! looking for shorter solutions.

use crate::cube::{Cube, Facelets, Mat};
use crate::cube3::Cube3;
use crate::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES, frame_rotation, rotate_move};
use crate::phase1::{P1State, P1Tables, Search1};
use crate::phase2::{P2Tables, Search2, standardise};
use crate::phase3::{P3Tables, Search3, Task};

/// Phase-3 checks at least this deep are split into one job per child of the root.
const SPLIT_DEPTH: u8 = 15;
use crate::table::Mod3Table;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub struct Tables {
    pub p1: P1Tables,
    pub p2: P2Tables,
    pub p3: P3Tables,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Stop after this long (the best solution so far is returned).
    pub time_limit: Duration,
    /// Stop as soon as a solution of at most this many moves is found.
    pub target_length: Option<usize>,
    /// Stop after this long even without a solution (the time limit is soft until there is
    /// one).
    pub hard_limit: Option<Duration>,
    pub threads: usize,
    /// Phase-1 endpoints collected in the first round (multiplied by 4 each round).
    pub endpoints: usize,
    /// Reductions collected in the first round (multiplied by 4 each round).
    pub reductions: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            time_limit: Duration::from_secs(5),
            target_length: None,
            hard_limit: None,
            threads: crate::par::threads(),
            endpoints: 128,
            reductions: 32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    /// Move indices (27-move set), in the frame of the input cube.
    pub moves: Vec<u8>,
    /// Lengths of the three phases (before merging moves at the phase boundaries).
    pub phases: [usize; 3],
    pub elapsed: Duration,
}

impl Solution {
    pub fn len(&self) -> usize {
        self.moves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub nodes1: u64,
    pub nodes2: u64,
    pub nodes3: u64,
    pub endpoints: usize,
    pub reductions: usize,
    pub rounds: usize,
    /// Wall time spent in each phase.
    pub time: [Duration; 3],
}

struct Candidate {
    moves1: Vec<u8>,
    frame: usize,
    cube: Cube,
    centre: u32,
    tau: [u8; 16],
    dt: u8,
    h2: u8,
    /// Phase-2 depths below this have been searched.
    done2: u8,
}

struct Reduction {
    moves12: Vec<u8>,
    len1: usize,
    frame: usize,
    c3: Cube3,
    h3: u8,
    /// Phase-3 depths below this have been searched without a solution.
    done3: u8,
}

/// The resumable state of one phase-1 length level.
struct Level1 {
    len: u8,
    jobs: Vec<Vec<u8>>,
    next: usize,
}

/// Merge consecutive moves of the same face (moves of one axis commute).
pub fn simplify(moves: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = vec![];
    for &m in moves {
        let (face, pow) = (m / 3, m % 3 + 1);
        let axis = face % 3;
        let mut merged = false;
        for j in (0..out.len()).rev() {
            let f = out[j] / 3;
            if f % 3 != axis {
                break;
            }
            if f == face {
                let p = (out[j] % 3 + 1 + pow) % 4;
                if p == 0 {
                    out.remove(j);
                } else {
                    out[j] = face * 3 + p - 1;
                }
                merged = true;
                break;
            }
        }
        if !merged {
            out.push(m);
        }
    }
    out
}

fn transpose(m: &Mat) -> Mat {
    let mut t = [[0i8; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = m[j][i];
        }
    }
    t
}

fn debug(start: Instant, msg: &str) {
    if std::env::var_os("EGGG_DEBUG").is_some() {
        eprintln!("[{:8.3}] {msg}", start.elapsed().as_secs_f64());
    }
}

pub fn verify(facelets: &Facelets, moves: &[u8]) -> bool {
    let end = moves.iter().fold(*facelets, |f, &m| {
        f.apply_turn(crate::moves::turn_of(m as usize))
    });
    end.is_solved()
}

/// Run `work(i)` for i in 0..n on `threads` threads, claiming indices in order.
fn par_jobs(n: usize, threads: usize, stop: &AtomicBool, work: impl Fn(usize) + Sync) {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.min(n).max(1) {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let j = next.fetch_add(1, Ordering::Relaxed);
                    if j >= n {
                        break;
                    }
                    work(j);
                }
            });
        }
    });
}

/// Phase-1 search below `srch.path` (state `st`, staging distance `d`, exactly `left` more
/// moves), expanding the first `split` levels here so that the job can be cut short: once
/// `close` is raised, each subtree not yet started at those levels is pushed to `spill` (as
/// its move prefix) instead of being searched. The subtrees visited, and the endpoints found,
/// are exactly those of `srch.dfs`.
#[allow(clippy::too_many_arguments)]
fn p1_run<F: FnMut(&[u8], u8) -> bool>(
    srch: &mut Search1<'_, F>,
    st: &P1State,
    d: u8,
    left: u8,
    last: usize,
    split: u8,
    close: &AtomicBool,
    spill: &mut Vec<Vec<u8>>,
) {
    if split == 0 || left <= 3 {
        srch.dfs(st, d, left, last);
        return;
    }
    let t = srch.t;
    let mut allowed = ALLOWED_AFTER[last];
    while allowed != 0 && !srch.stop {
        let m = allowed.trailing_zeros() as usize;
        allowed &= allowed - 1;
        let c = st.apply(m, t);
        let dc = Mod3Table::decode(d, t.value(&c));
        srch.nodes += 1;
        if dc >= left {
            continue;
        }
        srch.path.push(m as u8);
        if close.load(Ordering::Relaxed) {
            spill.push(srch.path.clone());
        } else {
            p1_run(srch, &c, dc, left - 1, m, split - 1, close, spill);
        }
        srch.path.pop();
    }
}

/// A phase-2 candidate from a phase-1 endpoint (None if its centres cannot be solved).
fn make_candidate(t: &Tables, cube: &Cube, moves1: &[u8], frame: usize) -> Option<Candidate> {
    let mut c = *cube;
    for &m in moves1 {
        c = c.apply_perm(&MOVE_PERMS[m as usize]);
    }
    let x = standardise(&c, frame);
    let centre = t.p2.centre_coord_of(&x);
    let cd = t.p2.centre_dist[centre as usize];
    if cd == crate::table::UNSEEN {
        return None;
    }
    let tau = t.p2.tau_of(&x);
    let dt = match &t.p2.joint {
        Some(j) => j.distance(&t.p2, centre / 1680, &tau, centre & 1),
        None => t.p2.tau_distance(&tau),
    };
    Some(Candidate {
        moves1: moves1.to_vec(),
        frame,
        cube: x,
        centre,
        tau,
        dt,
        h2: cd.max(dt),
        done2: 0,
    })
}

pub struct Solver<'a> {
    pub t: &'a Tables,
    pub opts: Options,
    pub stats: Stats,
}

struct Run<'a, 'r> {
    t: &'a Tables,
    cube: Cube,
    threads: usize,
    stop: &'r AtomicBool,
    /// Set once a solution exists (the time limit only applies from then on).
    have: &'r AtomicBool,
    start: Instant,
    best: Option<Solution>,
    report: &'r mut dyn FnMut(&Solution),
    target: Option<usize>,
    stats: Stats,
    root: P1State,
    d0: u8,
    level: Level1,
    seen_ep: HashSet<(Vec<u8>, usize)>,
    cands: Vec<Candidate>,
    reds: Vec<Reduction>,
    red_index: HashMap<(usize, Cube3), usize>,
}

impl<'a> Solver<'a> {
    pub fn new(t: &'a Tables, opts: Options) -> Self {
        Solver {
            t,
            opts,
            stats: Stats::default(),
        }
    }

    /// Solve `cube`, calling `report` with each improvement.
    pub fn solve(&mut self, cube: &Cube, report: &mut dyn FnMut(&Solution)) -> Option<Solution> {
        let start = Instant::now();
        let deadline = start + self.opts.time_limit;
        let hard = self.opts.hard_limit;
        let stop = AtomicBool::new(false);
        let have = AtomicBool::new(false);
        if cube.to_facelets().is_solved() {
            let s = Solution {
                moves: vec![],
                phases: [0; 3],
                elapsed: start.elapsed(),
            };
            report(&s);
            return Some(s);
        }
        let threads = self.opts.threads.max(1);
        let done = AtomicBool::new(false);
        let root = P1State::from_cube(cube);
        let d0 = self.t.p1.distance(&root);
        let mut run = Run {
            t: self.t,
            cube: *cube,
            threads,
            stop: &stop,
            start,
            have: &have,
            best: None,
            report,
            target: self.opts.target_length,
            stats: Stats::default(),
            root,
            d0,
            level: Level1 {
                len: d0,
                jobs: vec![],
                next: 0,
            },
            seen_ep: HashSet::new(),
            cands: vec![],
            reds: vec![],
            red_index: HashMap::new(),
        };
        run.level.jobs = run.level_jobs(d0);
        std::thread::scope(|scope| {
            // Watchdog that raises the stop flag at the deadline.
            let (stop_ref, done_ref, have_ref) = (&stop, &done, &have);
            scope.spawn(move || {
                while !done_ref.load(Ordering::Relaxed) {
                    // The limit is soft until there is a solution.
                    let now = Instant::now();
                    if (now >= deadline && have_ref.load(Ordering::Relaxed))
                        || hard.is_some_and(|h| now >= start + h)
                    {
                        stop_ref.store(true, Ordering::Relaxed);
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
            run.rounds(self.opts.endpoints, self.opts.reductions);
            if std::env::var_os("EGGG_DEBUG").is_some() {
                let mut h1 = std::collections::BTreeMap::new();
                for c in &run.cands {
                    *h1.entry((c.moves1.len(), c.moves1.len() + c.h2 as usize))
                        .or_insert(0usize) += 1;
                }
                let mut h12 = std::collections::BTreeMap::new();
                for r in &run.reds {
                    *h12.entry(r.moves12.len()).or_insert(0usize) += 1;
                }
                eprintln!("candidates by (L1, L1+h2): {h1:?}");
                eprintln!("reductions by L12: {h12:?}");
            }
            done.store(true, Ordering::Relaxed);
        });
        self.stats = run.stats;
        run.best
    }
}

impl<'a, 'r> Run<'a, 'r> {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn best_len(&self) -> usize {
        self.best.as_ref().map(|b| b.len()).unwrap_or(usize::MAX)
    }

    fn rounds(&mut self, endpoints: usize, reductions: usize) {
        let (mut want1, mut want2) = (endpoints.max(1), reductions.max(1));
        // Time spent producing reductions (phases 1 and 2) and checking them (phase 3).
        let (mut produce, mut check) = (Duration::ZERO, Duration::ZERO);
        loop {
            if self.stopped() {
                return;
            }
            self.stats.rounds += 1;
            let t0 = Instant::now();
            let tp = Instant::now();
            self.phase1(want1);
            self.stats.time[0] += tp.elapsed();
            debug(
                self.start,
                &format!(
                    "phase 1: {} endpoints (length {}), nodes {}",
                    self.cands.len(),
                    self.level.len,
                    self.stats.nodes1
                ),
            );
            let tp = Instant::now();
            self.phase2(want2);
            self.stats.time[1] += tp.elapsed();
            debug(
                self.start,
                &format!(
                    "phase 2: {} reductions, nodes {}",
                    self.reds.len(),
                    self.stats.nodes2
                ),
            );
            produce += t0.elapsed();
            // Phase 3 gets about 1.5 times the production time so far (shallow levels over the
            // whole pool are cheap; deep ones wait for a bigger pool).
            let t1 = Instant::now();
            let budget =
                (produce.mul_f64(std::env::var("EGGG_P3F").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5)).saturating_sub(check)).max(Duration::from_millis(10));
            self.phase3_levels(t1 + budget);
            self.stats.time[2] += t1.elapsed();
            check += t1.elapsed();
            if self.stopped() || self.level.len > 40 {
                return;
            }
            let g: usize = std::env::var("EGGG_GROW").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
            want1 *= g;
            want2 *= g;
        }
    }

    fn level_jobs(&self, len: u8) -> Vec<Vec<u8>> {
        let depth = if len < 3 {
            0
        } else if len < 7 {
            2
        } else {
            3
        };
        let mut jobs: Vec<Vec<u8>> = vec![vec![]];
        for _ in 0..depth {
            let mut next = vec![];
            for p in &jobs {
                let last = p.last().map(|&m| m as usize).unwrap_or(N_MOVES);
                let mut allowed = ALLOWED_AFTER[last];
                while allowed != 0 {
                    let m = allowed.trailing_zeros();
                    allowed &= allowed - 1;
                    let mut q = p.clone();
                    q.push(m as u8);
                    next.push(q);
                }
            }
            jobs = next;
        }
        // Keep only prefixes that can still reach the goal in `len` moves.
        let t = &self.t.p1;
        jobs.retain(|p| {
            let mut st = self.root;
            let mut d = self.d0;
            for (i, &m) in p.iter().enumerate() {
                st = st.apply(m as usize, t);
                d = Mod3Table::decode(d, t.value(&st));
                if d as usize > len as usize - i - 1 {
                    return false;
                }
            }
            true
        });
        jobs
    }

    /// Collect phase-1 endpoints until there are `want` candidates in total.
    fn phase1(&mut self, want: usize) {
        while self.cands.len() < want && !self.stopped() && self.level.len <= 40 {
            if self.level.next >= self.level.jobs.len() {
                let len = self.level.len + 1;
                self.level = Level1 {
                    len,
                    jobs: self.level_jobs(len),
                    next: 0,
                };
                continue;
            }
            let need = want - self.cands.len();
            let t = &self.t.p1;
            let (root, d0, len, stop) = (self.root, self.d0, self.level.len, self.stop);
            let jobs = &self.level.jobs;
            let first = self.level.next;
            let threads = self.threads;
            let claimed = AtomicUsize::new(first);
            let found = AtomicUsize::new(0);
            let nodes = AtomicU64::new(0);
            // Raised when enough endpoints are found, or when a quarter of the threads have
            // run out of jobs: running jobs then hand back their unstarted subtrees, so that no
            // thread waits long for the slowest job.
            let close = AtomicBool::new(false);
            let idle = AtomicUsize::new(0);
            let out = Mutex::new(vec![]);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut local = vec![];
                        let mut n = 0;
                        loop {
                            if stop.load(Ordering::Relaxed) || close.load(Ordering::Relaxed) {
                                break;
                            }
                            let j = claimed.fetch_add(1, Ordering::Relaxed);
                            if j >= jobs.len() {
                                if (idle.fetch_add(1, Ordering::Relaxed) + 1) * 4 >= threads {
                                    close.store(true, Ordering::Relaxed);
                                }
                                break;
                            }
                            let prefix = &jobs[j];
                            let mut st = root;
                            let mut d = d0;
                            for &m in prefix {
                                st = st.apply(m as usize, t);
                                d = Mod3Table::decode(d, t.value(&st));
                            }
                            let mut eps = vec![];
                            let mut spill = vec![];
                            let mut srch = Search1::new(t, |p: &[u8], f| {
                                eps.push((p.to_vec(), f));
                                true
                            });
                            srch.abort = Some(stop);
                            srch.path.extend_from_slice(prefix);
                            let last = prefix.last().map(|&m| m as usize).unwrap_or(N_MOVES);
                            let left = len - prefix.len() as u8;
                            p1_run(&mut srch, &st, d, left, last, 2, &close, &mut spill);
                            n += srch.nodes;
                            let k: usize = eps.iter().map(|e| e.1.count_ones() as usize).sum();
                            if found.fetch_add(k, Ordering::Relaxed) + k >= need {
                                close.store(true, Ordering::Relaxed);
                            }
                            local.push((j, eps, spill));
                        }
                        nodes.fetch_add(n, Ordering::Relaxed);
                        out.lock().unwrap().extend(local);
                    });
                }
            });
            self.stats.nodes1 += nodes.load(Ordering::Relaxed);
            let next = claimed.load(Ordering::Relaxed).min(jobs.len());
            let mut results = out.into_inner().unwrap();
            results.sort_by_key(|r| r.0);
            // Subtrees handed back go first, then the jobs not yet claimed.
            let mut rest: Vec<Vec<u8>> = vec![];
            for r in &mut results {
                rest.append(&mut r.2);
            }
            if rest.is_empty() {
                self.level.next = next;
            } else {
                rest.extend(self.level.jobs.drain(next..));
                self.level.jobs = rest;
                self.level.next = 0;
            }
            // New (endpoint, frame) pairs, in job order; their candidates are made in
            // parallel (standardising and the phase-2 bounds cost ~30 us each).
            let mut todo: Vec<(Vec<u8>, usize)> = vec![];
            for (_, eps, _) in results {
                for (moves, frames) in eps {
                    for f in 0..3 {
                        if frames >> f & 1 == 1 && self.seen_ep.insert((moves.clone(), f)) {
                            todo.push((moves.clone(), f));
                        }
                    }
                }
            }
            let made: Vec<Mutex<Option<Candidate>>> =
                (0..todo.len()).map(|_| Mutex::new(None)).collect();
            let (t, cube) = (self.t, &self.cube);
            crate::par::for_chunks(todo.len(), 64, |r| {
                for i in r {
                    *made[i].lock().unwrap() = make_candidate(t, cube, &todo[i].0, todo[i].1);
                }
            });
            self.cands
                .extend(made.into_iter().filter_map(|m| m.into_inner().unwrap()));
        }
        self.stats.endpoints = self.cands.len();
    }

    /// Phase 2, level by level on the reduction length, until `want` reductions in total.
    fn phase2(&mut self, want: usize) {
        // Reductions are only worth producing if a phase 3 of 14 moves or more could still
        // beat the best solution (14 moves: ~2e-4 of random 3x3 states, but that level has
        // ~10x the reductions of the one below, and phase 3 checks them cheaply).
        let cap = self.best_len().saturating_sub(15).min(64);
        let mut t12 = self
            .cands
            .iter()
            .map(|c| c.moves1.len() + c.h2.max(c.done2) as usize)
            .min()
            .unwrap_or(99);
        while self.reds.len() < want && !self.stopped() && t12 <= cap {
            for r in self.phase2_level(t12) {
                let key = (r.frame, r.c3);
                match self.red_index.get(&key) {
                    Some(&i) if self.reds[i].moves12.len() <= r.moves12.len() => {}
                    Some(&i) => self.reds[i] = r,
                    None => {
                        self.red_index.insert(key, self.reds.len());
                        self.reds.push(r);
                    }
                }
            }
            t12 += 1;
        }
        self.stats.reductions = self.reds.len();
    }

    /// Phase-2 searches of total length `t12` for every candidate that can reach it.
    fn phase2_level(&mut self, t12: usize) -> Vec<Reduction> {
        let t = self.t;
        let cands = &self.cands;
        let jobs: Vec<usize> = (0..cands.len())
            .filter(|&i| {
                let c = &cands[i];
                let l1 = c.moves1.len();
                t12 >= l1 + c.h2 as usize && (t12 - l1) as u8 >= c.done2
            })
            .collect();
        let out = Mutex::new(vec![]);
        let nodes = AtomicU64::new(0);
        let stop = self.stop;
        par_jobs(jobs.len(), self.threads, stop, |j| {
            let c = &cands[jobs[j]];
            let l2 = (t12 - c.moves1.len()) as u8;
            let rot = frame_rotation(c.frame);
            let rot_inv = transpose(&rot);
            let last1 = c
                .moves1
                .last()
                .map(|&m| rotate_move(m as usize, &rot))
                .unwrap_or(N_MOVES);
            let mut srch = Search2::new(&t.p2);
            srch.abort = Some(stop);
            let mut sols: Vec<Vec<u8>> = vec![];
            let mut keep = |p: &[u8]| {
                sols.push(p.to_vec());
                sols.len() < 64
            };
            if t.p2.joint.is_some() {
                srch.dfs_joint(c.centre, &c.tau, c.dt, l2, last1, &mut keep);
            } else {
                srch.dfs(c.centre, &c.tau, c.dt, l2, last1, &mut keep);
            }
            nodes.fetch_add(srch.nodes, Ordering::Relaxed);
            let mut local = vec![];
            for p in sols {
                let mut x = c.cube;
                for &m in &p {
                    x = x.apply_perm(&MOVE_PERMS[m as usize]);
                }
                let Some(c3) = Cube3::from_reduced(&x) else {
                    continue;
                };
                let mut moves12 = c.moves1.clone();
                moves12.extend(p.iter().map(|&m| rotate_move(m as usize, &rot_inv) as u8));
                let h3 = Search3::new(&t.p3).lower_bound(&c3);
                local.push(Reduction {
                    moves12,
                    len1: c.moves1.len(),
                    frame: c.frame,
                    c3,
                    h3,
                    done3: 0,
                });
            }
            out.lock().unwrap().push((j, local));
        });
        if !self.stopped() {
            for &j in &jobs {
                let c = &mut self.cands[j];
                c.done2 = c.done2.max((t12 - c.moves1.len()) as u8 + 1);
            }
        }
        self.stats.nodes2 += nodes.load(Ordering::Relaxed);
        let mut v = out.into_inner().unwrap();
        v.sort_by_key(|x| x.0);
        v.into_iter().flat_map(|x| x.1).collect()
    }

    fn record(&mut self, r: usize, sol3: &[u8]) {
        let red = &self.reds[r];
        let rot_inv = transpose(&frame_rotation(red.frame));
        let mut moves = red.moves12.clone();
        moves.extend(
            sol3.iter()
                .map(|&m| rotate_move(m as usize, &rot_inv) as u8),
        );
        let sol = Solution {
            moves: simplify(&moves),
            phases: [red.len1, red.moves12.len() - red.len1, sol3.len()],
            elapsed: self.start.elapsed(),
        };
        debug_assert!(verify(&self.cube.to_facelets(), &sol.moves));
        if sol.len() < self.best_len() {
            (self.report)(&sol);
            self.have.store(true, Ordering::Relaxed);
            self.best = Some(sol);
            if let Some(tl) = self.target
                && self.best_len() <= tl
            {
                self.stop.store(true, Ordering::Relaxed);
            }
        }
    }

    /// Phase 3, level by level on the total length, below the best solution, until the
    /// deadline (a level in progress then stops claiming reductions; finished ones are kept).
    fn phase3_levels(&mut self, deadline: Instant) {
        loop {
            let best = self.best_len();
            let lo = self
                .reds
                .iter()
                .map(|r| r.moves12.len() + r.h3.max(r.done3) as usize)
                .min()
                .unwrap_or(usize::MAX);
            if lo >= best || self.stopped() || Instant::now() >= deadline {
                return;
            }
            debug(
                self.start,
                &format!("phase 3: total {lo}, nodes {}", self.stats.nodes3),
            );
            match self.phase3_level(lo, deadline) {
                Some((r, sol)) => self.record(r, &sol),
                None => {
                    if Instant::now() >= deadline {
                        return;
                    }
                }
            }
        }
    }

    /// Phase-3 searches of total length `total` over the reductions; the first solution found.
    fn phase3_level(&mut self, total: usize, deadline: Instant) -> Option<(usize, Vec<u8>)> {
        let t = self.t;
        let reds = &self.reds;
        let mut order: Vec<usize> = (0..reds.len())
            .filter(|&i| {
                let r = &reds[i];
                let l12 = r.moves12.len();
                total >= l12 + r.h3 as usize && total - l12 >= r.done3 as usize
            })
            .collect();
        // Shallow searches first: any solution of this level is as good as any other.
        order.sort_by_key(|&i| (std::cmp::Reverse(reds[i].moves12.len()), i));
        // Deep searches are split into one task per child of the root, so that they spread
        // over the threads; `parts[r]` counts the jobs of reduction r.
        let mut jobs: Vec<(usize, Option<Task>)> = vec![];
        let mut parts = vec![0usize; reds.len()];
        let mut pruned = vec![];
        for &i in &order {
            let r = &reds[i];
            let d = (total - r.moves12.len()) as u8;
            if d >= SPLIT_DEPTH {
                match Search3::new(&t.p3).tasks(&r.c3, d) {
                    Some(ts) if !ts.is_empty() => {
                        parts[i] = ts.len();
                        jobs.extend(ts.into_iter().map(|task| (i, Some(task))));
                    }
                    // The root, or every child of it, is pruned: nothing to search.
                    _ => pruned.push(i),
                }
            } else {
                parts[i] = 1;
                jobs.push((i, None));
            }
        }
        let result = Mutex::new(None::<(usize, Vec<u8>)>);
        let found = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let finished: Vec<AtomicUsize> = (0..reds.len()).map(|_| AtomicUsize::new(0)).collect();
        let stop = self.stop;
        par_jobs(jobs.len(), self.threads, stop, |j| {
            if found.load(Ordering::Relaxed) || Instant::now() >= deadline {
                return;
            }
            let (i, task) = &jobs[j];
            let r = &reds[*i];
            let d = (total - r.moves12.len()) as u8;
            let mut srch = Search3::new(&t.p3);
            srch.abort = Some(stop);
            srch.abort_level = Some(&found);
            let sol = match task {
                Some(task) => srch.run_task(task),
                None => srch.search_depth(&r.c3, d, N_MOVES),
            };
            nodes.fetch_add(srch.nodes, Ordering::Relaxed);
            match sol {
                Some(sol) => {
                    found.store(true, Ordering::Relaxed);
                    let mut g = result.lock().unwrap();
                    if g.as_ref().is_none_or(|(k, _)| *i < *k) {
                        *g = Some((*i, sol));
                    }
                }
                None if !srch.aborted => {
                    finished[*i].fetch_add(1, Ordering::Relaxed);
                }
                None => {}
            }
        });
        self.stats.nodes3 += nodes.load(Ordering::Relaxed);
        for i in 0..self.reds.len() {
            if (parts[i] > 0 && finished[i].load(Ordering::Relaxed) == parts[i])
                || pruned.contains(&i)
            {
                let r = &mut self.reds[i];
                r.done3 = r.done3.max((total - r.moves12.len()) as u8 + 1);
            }
        }
        result.into_inner().unwrap()
    }
}

/// Load or build the phase-1 tables in `dir`.
pub fn load_p1(dir: &std::path::Path, threads: usize, verbose: bool) -> P1Tables {
    let big = p1_big();
    let (p1_name, p1_len) = if big {
        ("p1-staging-big", crate::phase1::Staging::big_len())
    } else {
        ("p1-staging2", P1Tables::table_len())
    };
    let p1_path = dir.join(format!("{p1_name}.tab"));
    let staging = crate::phase1::Staging::new();
    let n = p1_len;
    let mut table = Mod3Table {
        data: crate::table::HugeBuf::zeroed(n.div_ceil(32)),
        n: n as u64,
    };
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            table.data.as_mut_slice().as_mut_ptr() as *mut u8,
            table.data.byte_len(),
        )
    };
    if crate::table::load(&p1_path, p1_name, bytes).is_err() {
        if verbose {
            eprintln!("generating phase-1 staging table ({p1_name})...");
        }
        let depths = if big {
            staging.generate_big(threads)
        } else {
            staging.generate(threads)
        };
        table = Mod3Table::from_depths(&depths);
        let bytes = unsafe {
            std::slice::from_raw_parts(
                table.data.as_slice().as_ptr() as *const u8,
                table.data.byte_len(),
            )
        };
        crate::table::save(&p1_path, p1_name, bytes).expect("saving table");
    }
    let mut p1 = P1Tables::with_table(staging, table);
    p1.big = big;
    p1
}

/// Load or build all tables in `dir`.
pub fn load_tables(dir: &std::path::Path, h48_k: u32, threads: usize, verbose: bool) -> Tables {
    load_tables_with(dir, h48_k, false, threads, verbose)
}

/// Large phase-1 staging table instead of the symmetry-reduced one ($EGGG_P1_BIG, set by
/// `--p1-big`).
fn p1_big() -> bool {
    std::env::var_os("EGGG_P1_BIG").is_some()
}

/// Load or build all tables in `dir`; `joint`: also the joint phase-2 table (2.1 GB).
pub fn load_tables_with(
    dir: &std::path::Path,
    h48_k: u32,
    joint: bool,
    threads: usize,
    verbose: bool,
) -> Tables {
    let t0 = Instant::now();
    let p1 = load_p1(dir, threads, verbose);
    let p2_path = dir.join("p2-tau.tab");
    let p2 = {
        let n = crate::phase2::N_TAU;
        let mut table = Mod3Table {
            data: crate::table::HugeBuf::zeroed(n.div_ceil(32)),
            n: n as u64,
        };
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                table.data.as_mut_slice().as_mut_ptr() as *mut u8,
                table.data.byte_len(),
            )
        };
        if crate::table::load(&p2_path, "p2-tau", bytes).is_ok() {
            P2Tables::build(Some(table), threads)
        } else {
            if verbose {
                eprintln!("generating phase-2 pairing table...");
            }
            let t = P2Tables::build(None, threads);
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    t.tau_table.data.as_slice().as_ptr() as *const u8,
                    t.tau_table.data.byte_len(),
                )
            };
            crate::table::save(&p2_path, "p2-tau", bytes).expect("saving table");
            t
        }
    };
    let mut p2 = p2;
    if joint {
        let mut j = crate::p2joint::Joint::new(&p2);
        let path = dir.join("p2-joint2.tab");
        j.table = crate::table::HugeBuf::zeroed(j.bytes() / 8);
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                j.table.as_mut_slice().as_mut_ptr() as *mut u8,
                j.bytes(),
            )
        };
        if crate::table::load(&path, "p2-joint2", bytes).is_err() {
            if verbose {
                eprintln!(
                    "generating the joint phase-2 table ({:.1} GB)...",
                    j.bytes() as f64 / 1e9
                );
            }
            j.generate(&p2, threads, verbose);
            let bytes = unsafe {
                std::slice::from_raw_parts(j.table.as_slice().as_ptr() as *const u8, j.bytes())
            };
            crate::table::save(&path, "p2-joint2", bytes).expect("saving table");
        }
        p2.joint = Some(j);
    }
    let p3 = P3Tables::load_or_build(dir, h48_k, threads, verbose);
    // Huge pages for the big tables, in parallel.
    std::thread::scope(|s| {
        s.spawn(|| p1.table.data.collapse());
        s.spawn(|| p2.tau_table.data.collapse());
        if let Some(j) = &p2.joint {
            s.spawn(|| j.table.collapse());
        }
        s.spawn(|| p3.h48.data.collapse());
    });
    if verbose {
        eprintln!("tables ready in {:.1} s", t0.elapsed().as_secs_f64());
    }
    Tables { p1, p2, p3 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Jobs cut short at random points, with their handed-back subtrees searched later, find
    /// exactly the endpoints of one uninterrupted search.
    #[test]
    fn split_jobs_find_the_same_endpoints() {
        let t = load_p1(std::path::Path::new("tables"), 8, false);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        for scramble in 0..6 {
            let mut cube = Cube::SOLVED;
            for _ in 0..9 {
                let m = rand::Rng::gen_range(&mut rng, 0..N_MOVES);
                cube = cube.apply_perm(&MOVE_PERMS[m]);
            }
            let root = P1State::from_cube(&cube);
            let d0 = t.distance(&root);
            for len in d0.max(4)..d0.max(4) + 2 {
                let mut want = vec![];
                let mut srch = Search1::new(&t, |p: &[u8], f| {
                    want.push((p.to_vec(), f));
                    true
                });
                srch.dfs(&root, d0, len, N_MOVES);
                drop(srch);
                want.sort();

                let mut got = vec![];
                let mut queue: Vec<Vec<u8>> = vec![vec![]];
                let mut count = 0usize;
                while let Some(prefix) = queue.pop() {
                    let close = AtomicBool::new(false);
                    let mut st = root;
                    let mut d = d0;
                    for &m in &prefix {
                        st = st.apply(m as usize, &t);
                        d = Mod3Table::decode(d, t.value(&st));
                    }
                    let mut spill = vec![];
                    let mut srch = Search1::new(&t, |p: &[u8], f| {
                        got.push((p.to_vec(), f));
                        count += 1;
                        if count % 3 == scramble % 3 {
                            close.store(true, Ordering::Relaxed);
                        }
                        true
                    });
                    srch.path.extend_from_slice(&prefix);
                    let last = prefix.last().map(|&m| m as usize).unwrap_or(N_MOVES);
                    p1_run(
                        &mut srch,
                        &st,
                        d,
                        len - prefix.len() as u8,
                        last,
                        2,
                        &close,
                        &mut spill,
                    );
                    drop(srch);
                    queue.extend(spill);
                }
                got.sort();
                assert!(
                    !want.is_empty() || len == d0,
                    "no endpoints at length {len}"
                );
                assert_eq!(got, want, "scramble {scramble}, length {len}");
            }
        }
    }
}
