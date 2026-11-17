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
use crate::phase3::{P3Tables, Search3};
use crate::table::Mod3Table;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Deepest phase-3 check (once there is a solution).
const DMAX: usize = 17;
/// Reductions are produced only if they could lead to a solution with a phase 3 this short.
const DMIN: usize = 15;

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
    red_index: HashMap<Cube3, usize>,
    /// Widening of the caps before the first solution.
    slack: usize,
    /// Shortest phase-3 length a produced reduction is expected to need.
    dmin: usize,
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
            slack: 0,
            dmin: DMIN,
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
            run.run(self.opts.endpoints);
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

    /// Lowest reduction length in the pool.
    fn lstar(&self) -> Option<usize> {
        self.reds.iter().map(|r| r.moves12.len()).min()
    }

    /// Target total of the phase-3 checks, and the deepest check allowed.
    fn check_target(&self) -> (usize, usize) {
        match &self.best {
            Some(b) => (b.len() - 1, DMAX),
            // Before the first solution every reduction may be checked to the depth cap; the
            // lowest totals come first.
            None => (usize::MAX, DMAX + self.slack),
        }
    }

    /// Longest reductions worth producing.
    fn production_cap(&self) -> usize {
        match (&self.best, self.lstar()) {
            (Some(b), _) => (b.len() - 1).saturating_sub(self.dmin),
            (None, Some(l)) => l + 1 + self.slack,
            (None, None) => usize::MAX,
        }
    }

    /// Next phase-3 level: the lowest total at which some reduction has an open check.
    fn next_check(&self) -> Option<usize> {
        let (target, dmax) = self.check_target();
        self.reds
            .iter()
            .filter_map(|r| {
                let l = r.moves12.len();
                let d = r.h3.max(r.done3) as usize;
                (l + d <= target && d <= dmax).then_some(l + d)
            })
            .min()
    }

    /// Next phase-2 level: the lowest reduction length some candidate can still be searched at.
    fn next_production(&self) -> Option<usize> {
        let cap = self.production_cap();
        self.cands
            .iter()
            .map(|c| c.moves1.len() + c.h2.max(c.done2) as usize)
            .filter(|&t| t <= cap)
            .min()
    }

    /// The scheduler: phase-3 checks first (they find solutions for far fewer nodes than
    /// producing reductions does), then phase-2 searches below the production cap, then more
    /// phase-1 endpoints.
    fn run(&mut self, batch1: usize) {
        loop {
            if self.stopped() {
                return;
            }
            if let Some(t) = self.next_check() {
                debug(
                    self.start,
                    &format!(
                        "phase 3: total {t} ({} reductions), nodes {}",
                        self.reds.len(),
                        self.stats.nodes3
                    ),
                );
                if let Some((r, sol)) = self.phase3_level(t) {
                    self.record(r, &sol);
                }
                continue;
            }
            if let Some(t) = self.next_production() {
                let before = self.reds.len();
                self.phase2_batch(t);
                debug(
                    self.start,
                    &format!(
                        "phase 2: level {t}: {} new reductions ({} total), nodes {}",
                        self.reds.len() - before,
                        self.reds.len(),
                        self.stats.nodes2
                    ),
                );
                continue;
            }
            let want = batch1.max(self.cands.len() / 4);
            let cap = self.production_cap();
            let useful = self.phase1(want);
            debug(
                self.start,
                &format!(
                    "phase 1: {} endpoints (length {}), {useful} under cap {cap}, nodes {}",
                    self.cands.len(),
                    self.level.len,
                    self.stats.nodes1
                ),
            );
            if useful == 0 {
                // Nothing new can help within the caps: widen them.
                if self.best.is_none() {
                    self.slack += 1;
                } else if self.dmin > 10 {
                    self.dmin -= 1;
                } else {
                    return;
                }
            }
            if self.level.len > 40 {
                return;
            }
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

    /// Collect about `want` more phase-1 endpoints; returns how many of the new candidates can
    /// reach the production cap.
    fn phase1(&mut self, want: usize) -> usize {
        let want = self.cands.len() + want;
        let first_new = self.cands.len();
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
            let claimed = AtomicUsize::new(first);
            let found = AtomicUsize::new(0);
            let nodes = AtomicU64::new(0);
            let out = Mutex::new(vec![]);
            std::thread::scope(|s| {
                for _ in 0..self.threads {
                    s.spawn(|| {
                        let mut local = vec![];
                        let mut n = 0;
                        while !stop.load(Ordering::Relaxed) && found.load(Ordering::Relaxed) < need
                        {
                            let j = claimed.fetch_add(1, Ordering::Relaxed);
                            if j >= jobs.len() {
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
                            let mut srch = Search1::new(t, |p: &[u8], f| {
                                eps.push((p.to_vec(), f));
                                true
                            });
                            srch.abort = Some(stop);
                            srch.path.extend_from_slice(prefix);
                            let last = prefix.last().map(|&m| m as usize).unwrap_or(N_MOVES);
                            srch.dfs(&st, d, len - prefix.len() as u8, last);
                            n += srch.nodes;
                            found.fetch_add(
                                eps.iter().map(|e| e.1.count_ones() as usize).sum(),
                                Ordering::Relaxed,
                            );
                            local.push((j, eps));
                        }
                        nodes.fetch_add(n, Ordering::Relaxed);
                        out.lock().unwrap().extend(local);
                    });
                }
            });
            self.stats.nodes1 += nodes.load(Ordering::Relaxed);
            self.level.next = claimed.load(Ordering::Relaxed).min(jobs.len());
            let mut results = out.into_inner().unwrap();
            results.sort_by_key(|r| r.0);
            for (_, eps) in results {
                for (moves, frames) in eps {
                    for f in 0..3 {
                        if frames >> f & 1 == 1
                            && self.seen_ep.insert((moves.clone(), f))
                            && let Some(c) = self.make_candidate(&moves, f)
                        {
                            self.cands.push(c);
                        }
                    }
                }
            }
        }
        self.stats.endpoints = self.cands.len();
        let cap = self.production_cap();
        self.cands[first_new..]
            .iter()
            .filter(|c| c.moves1.len() + c.h2 as usize <= cap)
            .count()
    }

    fn make_candidate(&self, moves1: &[u8], frame: usize) -> Option<Candidate> {
        let t = self.t;
        let mut c = self.cube;
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
        let dt = t.p2.tau_distance(&tau);
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

    /// Phase-2 searches at reduction length `t`, for a batch of the candidates whose next
    /// open level it is (the cheapest-looking first), adding the reductions to the pool.
    fn phase2_batch(&mut self, t: usize) {
        let mut jobs: Vec<usize> = (0..self.cands.len())
            .filter(|&i| {
                let c = &self.cands[i];
                c.moves1.len() + c.h2.max(c.done2) as usize == t
            })
            .collect();
        jobs.sort_by_key(|&i| {
            (
                t - self.cands[i].moves1.len() - self.cands[i].h2 as usize,
                i,
            )
        });
        jobs.truncate((self.threads * 16).max(64));
        let target = self.check_target().0;
        for r in self.phase2_jobs(&jobs, t) {
            if self.best.is_some() && r.moves12.len() + r.h3 as usize > target {
                continue;
            }
            match self.red_index.get(&r.c3) {
                Some(&i) if self.reds[i].moves12.len() <= r.moves12.len() => {}
                Some(&i) => self.reds[i] = r,
                None => {
                    self.red_index.insert(r.c3, self.reds.len());
                    self.reds.push(r);
                }
            }
        }
        self.stats.reductions = self.reds.len();
    }

    /// Phase-2 searches of total length `t12` for the given candidates.
    fn phase2_jobs(&mut self, jobs: &[usize], t12: usize) -> Vec<Reduction> {
        let t = self.t;
        let cands = &self.cands;
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
            srch.dfs(c.centre, &c.tau, c.dt, l2, last1, &mut |p: &[u8]| {
                sols.push(p.to_vec());
                sols.len() < 4096
            });
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
            out.lock()
                .unwrap()
                .push((j, local, !stop.load(Ordering::Relaxed)));
        });
        let mut v = out.into_inner().unwrap();
        for &(j, _, complete) in &v {
            if complete {
                let c = &mut self.cands[jobs[j]];
                c.done2 = c.done2.max((t12 - c.moves1.len()) as u8 + 1);
            }
        }
        self.stats.nodes2 += nodes.load(Ordering::Relaxed);
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

    /// Phase-3 searches of total length `total` over the reductions; the first solution found.
    fn phase3_level(&mut self, total: usize) -> Option<(usize, Vec<u8>)> {
        let t = self.t;
        let reds = &self.reds;
        let dmax = self.check_target().1;
        let mut jobs: Vec<usize> = (0..reds.len())
            .filter(|&i| {
                let r = &reds[i];
                let l12 = r.moves12.len();
                l12 + r.h3.max(r.done3) as usize == total && total - l12 <= dmax
            })
            .collect();
        // Shallow searches first: any solution of this level is as good as any other.
        jobs.sort_by_key(|&i| (std::cmp::Reverse(reds[i].moves12.len()), i));
        let result = Mutex::new(None::<(usize, Vec<u8>)>);
        let found = AtomicBool::new(false);
        let nodes = AtomicU64::new(0);
        let finished = Mutex::new(vec![]);
        let stop = self.stop;
        par_jobs(jobs.len(), self.threads, stop, |j| {
            if found.load(Ordering::Relaxed) {
                return;
            }
            let r = &reds[jobs[j]];
            let d = (total - r.moves12.len()) as u8;
            let mut srch = Search3::new(&t.p3);
            srch.abort = Some(stop);
            srch.abort_level = Some(&found);
            let sol = srch.search_depth(&r.c3, d, N_MOVES);
            nodes.fetch_add(srch.nodes, Ordering::Relaxed);
            match sol {
                Some(sol) => {
                    found.store(true, Ordering::Relaxed);
                    let mut g = result.lock().unwrap();
                    if g.as_ref().is_none_or(|(k, _)| jobs[j] < *k) {
                        *g = Some((jobs[j], sol));
                    }
                }
                None if !srch.aborted => finished.lock().unwrap().push(jobs[j]),
                None => {}
            }
        });
        self.stats.nodes3 += nodes.load(Ordering::Relaxed);
        for j in finished.into_inner().unwrap() {
            let r = &mut self.reds[j];
            r.done3 = r.done3.max((total - r.moves12.len()) as u8 + 1);
        }
        result.into_inner().unwrap()
    }
}

/// Load or build all tables in `dir`.
pub fn load_tables(dir: &std::path::Path, h48_k: u32, threads: usize, verbose: bool) -> Tables {
    let t0 = Instant::now();
    let p1_path = dir.join("p1-staging2.tab");
    let p1 = {
        let staging = crate::phase1::Staging::new();
        let n = P1Tables::table_len();
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
        if crate::table::load(&p1_path, "p1-staging2", bytes).is_err() {
            if verbose {
                eprintln!("generating phase-1 staging table...");
            }
            let depths = staging.generate(threads);
            table = Mod3Table::from_depths(&depths);
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    table.data.as_slice().as_ptr() as *const u8,
                    table.data.byte_len(),
                )
            };
            crate::table::save(&p1_path, "p1-staging2", bytes).expect("saving table");
        }
        P1Tables::with_table(staging, table)
    };
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
    let p3 = P3Tables::load_or_build(dir, h48_k, threads, verbose);
    if verbose {
        eprintln!("tables ready in {:.1} s", t0.elapsed().as_secs_f64());
    }
    Tables { p1, p2, p3 }
}
