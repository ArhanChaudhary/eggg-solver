//! The three-phase driver.
//!
//! 1. Phase 1 collects endpoints by increasing length (one search tree serves all 18
//!    frame/colour-assignment combinations); each length level is split into jobs and can be
//!    resumed, so no endpoint is searched twice.
//! 2. Each endpoint is brought into the standard frame, and phase 2 is searched level by level
//!    on the reduction length, collecting reductions (deduplicated by their 3x3x3).
//! 3. Phase 3 searches level by level on the total length over all reductions (an IDA*
//!    per reduction, split into tasks over the threads), so each solution found is the
//!    shortest the pool allows; phase 2 then only produces reductions that could beat it.
//!
//! Rounds repeat with geometrically more endpoints and reductions while time remains, only
//! looking for shorter solutions.

use crate::cube::{Cube, Facelets, Mat};
use crate::cube3::Cube3;
use crate::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES, frame_rotation, rotate_move};
use crate::phase1::{P1State, P1Tables, Search1};
use crate::phase2::{P2Tables, Search2, Task2, first_moves, standardise};
use crate::phase3::{P3Tables, Search3, Task};

/// Phase-3 checks at least this deep are split into one job per child of the root.
const SPLIT_DEPTH: u8 = 15;
use crate::table::Mod3Table;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The deepest phase-2 level is best - DEEPEST_MARGIN (see `Run::phase2`).
const DEEPEST_MARGIN: usize = 13;

/// Once a solution exists, a round that would run past the time limit is cut short: its
/// phase 1 stops after this share of the time left when the round starts, and phase 2 stops
/// after `P2_SHARE` of it, so that phases 2 and 3 still search the round's endpoints (see
/// `Run::rounds`).
const P1_SHARE: f64 = 1.0 / 1.3;
const P2_SHARE: f64 = 0.96;

/// Once a solution exists, each round's phase 1 lasts at most this share of the time elapsed
/// (but at least `MIN_ROUND`): rounds then grow ~1.25x instead of 2x, so that the endpoints
/// found are searched by phases 2 and 3 sooner.
const ROUND_SHARE: f64 = 0.25;
const MIN_ROUND: Duration = Duration::from_millis(20);

/// Once a solution exists, a phase-2 search whose depth exceeds its bound by at least this
/// much is split into tasks one move below its root (two moves if by more): such searches take
/// ~5 ms to 1.5 s on the M2, the others less than a millisecond (see `Run::phase2_level`).
const P2_SPLIT_SLACK: u8 = 3;

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
    /// Phase-1 endpoints collected in the first round (doubled each round).
    pub endpoints: usize,
    /// Reductions collected in the first round (doubled each round).
    pub reductions: usize,
    /// Once a solution exists, search each new phase-1 length's jobs best first, by their
    /// prefix's yield at the previous length (see [`order_by_yield`]); otherwise in move
    /// order.
    pub order_jobs: bool,
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
            order_jobs: true,
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
    /// Exact tau distance, or with the joint table, the exact joint distance.
    dt: u8,
    /// With the joint table, the exact joint distance of the F/B view (else 0).
    df: u8,
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
    /// (colour pair, move prefix): see [`Search1::pair`].
    jobs: Vec<(u8, Vec<u8>)>,
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

/// Tells the phase-1 jobs of a batch to hand back their unstarted subtrees: raised when enough
/// endpoints are found or threads run out of jobs, and at time `at` (the round's phase-1 cut).
struct Close {
    flag: AtomicBool,
    at: Option<Instant>,
}

impl Close {
    fn new(at: Option<Instant>) -> Self {
        Close {
            flag: AtomicBool::new(false),
            at,
        }
    }

    fn raise(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    fn is_raised(&self) -> bool {
        if self.flag.load(Ordering::Relaxed) {
            return true;
        }
        if self.at.is_some_and(|t| Instant::now() >= t) {
            self.raise();
            return true;
        }
        false
    }
}

/// Phase-1 search below `srch.path` (state `st`, staging distance `d`, exactly `left` more
/// moves), expanding the first `split` levels here so that the job can be cut short: once
/// `close` is raised, each subtree not yet started at those levels is pushed to `spill` (as
/// its move prefix) instead of being searched. The subtrees visited, and the endpoints found,
/// are exactly those of `srch.dfs`. The nodes of each subtree whose path reaches 4 moves here
/// are added to `per4` (by that 4-move prefix; see [`Run::yields`]): as many as a job of that
/// prefix counts.
#[allow(clippy::too_many_arguments)]
fn p1_run<F: FnMut(&[u8], u8) -> bool>(
    srch: &mut Search1<'_, F>,
    st: &P1State,
    d: u8,
    left: u8,
    last: usize,
    split: u8,
    close: &Close,
    spill: &mut Vec<Vec<u8>>,
    per4: &mut Vec<([u8; 4], u64)>,
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
        let dc = Mod3Table::decode(d, srch.value(&c));
        srch.nodes += 1;
        if dc >= left {
            continue;
        }
        let n0 = srch.nodes;
        srch.path.push(m as u8);
        if close.is_raised() {
            spill.push(srch.path.clone());
        } else {
            p1_run(srch, &c, dc, left - 1, m, split - 1, close, spill, per4);
        }
        if let Some(k) = prefix4(&srch.path).filter(|_| srch.path.len() == 4) {
            per4.push((k, srch.nodes - n0));
        }
        srch.path.pop();
    }
}

/// Phase-2 candidates from phase-1 endpoints `(moves, frame bits)`, appended to `out` in order
/// (by endpoint, then frame), except the frames whose centres cannot be solved and those whose
/// phase-2 bound h2 puts L1 + h2 past `limit`. The bound is exact for the candidates kept; for
/// the others, walking down the joint table stops once it is out of reach. The joint-table
/// walks are interleaved (each is a chain of dependent table reads); the F/B views are walked
/// only for the frames within reach after the U/D walk.
fn make_candidates(
    p2: &P2Tables,
    cube: &Cube,
    eps: &[(Vec<u8>, u8)],
    limit: usize,
    out: &mut Vec<Candidate>,
) {
    // (endpoint, frame, standardised cube, centre coordinate, centre distance) of the frames
    // within the centre bound, and their joint-table walks (U/D pattern, tau, PLL bit, bound).
    let mut part = Vec::with_capacity(eps.len());
    let mut walks = Vec::with_capacity(eps.len());
    // The cube after each prefix of the last endpoint (consecutive endpoints of a search share
    // most of their moves).
    let mut path = vec![*cube];
    let mut prev: &[u8] = &[];
    for (e, (moves1, frames)) in eps.iter().enumerate() {
        let common = prev.iter().zip(moves1).take_while(|(a, b)| a == b).count();
        path.truncate(common + 1);
        for &m in &moves1[common..] {
            let c = path[path.len() - 1].apply_perm(&MOVE_PERMS[m as usize]);
            path.push(c);
        }
        prev = moves1;
        let c = path[moves1.len()];
        for frame in 0..3 {
            if frames >> frame & 1 == 0 {
                continue;
            }
            let x = standardise(&c, frame);
            let centre = p2.centre_coord_of(&x);
            let cd = p2.centre_dist[centre as usize];
            if cd == crate::table::UNSEEN || moves1.len() + cd as usize > limit {
                continue;
            }
            let room = (limit - moves1.len()).min(u8::MAX as usize) as u8;
            part.push((e, frame, x, centre, cd));
            walks.push((centre / 1680, p2.tau_of(&x), centre & 1, room));
        }
    }
    let mut dts = Vec::with_capacity(walks.len());
    let mut dfs = vec![Some(0); walks.len()];
    match &p2.joint {
        Some(j) => {
            j.distances_at_most(&walks, &mut dts);
            // The F/B views of the frames still within reach.
            let (mut keep, mut views) = (vec![], vec![]);
            for (k, (w, dt)) in walks.iter().zip(&dts).enumerate() {
                if dt.is_some() {
                    keep.push(k);
                    let centre = part[k].3;
                    views.push((j.fb.ud(centre), j.fb.tau(&w.1), w.2, w.3));
                }
            }
            let mut got = Vec::with_capacity(views.len());
            j.distances_at_most(&views, &mut got);
            dfs.fill(None);
            for (k, d) in keep.into_iter().zip(got) {
                dfs[k] = d;
            }
        }
        None => dts.extend(walks.iter().map(|w| Some(p2.tau_distance(&w.1)))),
    }
    for (((&(e, frame, x, centre, cd), w), dt), df) in part.iter().zip(&walks).zip(dts).zip(dfs) {
        let moves1 = &eps[e].0;
        let (Some(dt), Some(df)) = (dt, df) else {
            continue;
        };
        let h2 = cd.max(dt).max(df);
        if moves1.len() + h2 as usize > limit {
            continue;
        }
        out.push(Candidate {
            moves1: moves1.clone(),
            frame,
            cube: x,
            centre,
            tau: w.1,
            dt,
            df,
            h2,
            done2: 0,
        });
    }
}

/// The phase-2 bound h2 of each candidate the driver would make from phase-1 endpoints `eps`
/// (see [`make_candidates`]: frames whose L1 + h2 is within `limit`): (endpoint index, frame,
/// h2), in endpoint order. For analysis tools.
pub fn candidate_bounds(
    p2: &P2Tables,
    cube: &Cube,
    eps: &[(Vec<u8>, u8)],
    limit: usize,
) -> Vec<(usize, usize, u8)> {
    let mut cands = vec![];
    make_candidates(p2, cube, eps, limit, &mut cands);
    let mut e = 0;
    cands
        .iter()
        .map(|c| {
            while eps[e].0 != c.moves1 {
                e += 1;
            }
            (e, c.frame, c.h2)
        })
        .collect()
}

/// The first four moves of a phase-1 path (the key of the driver's yield statistics).
fn prefix4(moves: &[u8]) -> Option<[u8; 4]> {
    moves.get(..4).map(|p| [p[0], p[1], p[2], p[3]])
}

/// Reorder a new phase-1 level's jobs best first, by the yield of their prefix at the previous
/// length (`yields`: kept candidates and phase-1 nodes per 4-move prefix): kept candidates per
/// node, estimated hierarchically. A 3-move prefix's rate is smoothed towards the mean over
/// all prefixes by one mean-sized 3-move prefix's worth of nodes, and a 4-move prefix's
/// towards its 3-move prefix's estimate by one mean-sized 4-move prefix's worth (unknown
/// prefixes get their parent's estimate); a job is ranked by the estimate of its prefix's
/// first 4 moves (3 if it has only 3). Ties keep their order. The jobs are only permuted.
///
/// A job's yield at one length predicts its yield at the next (its first moves fix much of
/// the centres' and wings' structure): sampled on 4 cubes, the quarter of the length-16
/// jobs with the best length-15 yield held 60% of the length-16 kept candidates in 45% of the
/// nodes. Only the last length a run reaches is searched partly, so only there does the
/// order matter: its best jobs come first. On 20 bench positions (examples/p1yield, each
/// cube's length cut short by the time limit, predicted from the length before), ranking
/// 4-move prefixes instead of 3-move ones puts 0.47 instead of 0.42 of the kept candidates in
/// the first 20% of the nodes (0.34 instead of 0.28 in the first 10%).
fn order_by_yield(jobs: &mut Vec<(u8, Vec<u8>)>, yields: &HashMap<[u8; 4], (u64, u64)>) {
    if yields.is_empty() {
        return;
    }
    let mut y3: HashMap<[u8; 3], (u64, u64)> = HashMap::new();
    for (k, v) in yields {
        let e = y3.entry([k[0], k[1], k[2]]).or_default();
        e.0 += v.0;
        e.1 += v.1;
    }
    let (kept, nodes) = y3
        .values()
        .fold((0u64, 0u64), |a, v| (a.0 + v.0, a.1 + v.1));
    let mean3 = (nodes as f64 / y3.len() as f64).max(1.0);
    let mean4 = (nodes as f64 / yields.len() as f64).max(1.0);
    let rate = kept as f64 / (nodes as f64).max(1.0);
    let score = |p: &[u8]| {
        if p.len() < 3 {
            return rate;
        }
        let (k, n) = y3.get(&[p[0], p[1], p[2]]).copied().unwrap_or((0, 0));
        let s3 = (k as f64 + rate * mean3) / (n as f64 + mean3);
        match prefix4(p) {
            None => s3,
            Some(key) => {
                let (k, n) = yields.get(&key).copied().unwrap_or((0, 0));
                (k as f64 + s3 * mean4) / (n as f64 + mean4)
            }
        }
    };
    let mut order: Vec<(f64, usize)> = jobs
        .iter()
        .enumerate()
        .map(|(i, j)| (score(&j.1), i))
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut old: Vec<Option<(u8, Vec<u8>)>> = jobs.drain(..).map(Some).collect();
    jobs.extend(order.iter().map(|&(_, i)| old[i].take().unwrap()));
}

/// The frame rotation of a candidate's phase 2, the last phase-1 move in that frame, and the
/// moves allowed to open phase 2 (see [`first_moves`]).
fn phase2_root(c: &Candidate) -> (Mat, usize, u32) {
    let rot = frame_rotation(c.frame);
    let last1 = c
        .moves1
        .last()
        .map(|&m| rotate_move(m as usize, &rot))
        .unwrap_or(N_MOVES);
    // Phase 2 may open with moves of phase 1's trailing axis (on its unused faces).
    let tail: Vec<u8> = c.moves1[c.moves1.len().saturating_sub(3)..]
        .iter()
        .map(|&m| rotate_move(m as usize, &rot) as u8)
        .collect();
    (rot, last1, first_moves(&tail))
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
    /// When the run stops once a solution exists (the earlier of the time and hard limits).
    limit: Instant,
    /// This round's phase 1 stops at `p1_cut`, and phase 2 at `p2_cut` (None: no cut).
    p1_cut: Option<Instant>,
    p2_cut: Option<Instant>,
    /// Phase-2 searches with at least this slack are split (`P2_SPLIT_SLACK`).
    split2: u8,
    best: Option<Solution>,
    report: &'r mut dyn FnMut(&Solution),
    target: Option<usize>,
    stats: Stats,
    root: P1State,
    /// The root's exact distance in each pair's pattern table and (index 3) the staging table.
    d0: [u8; 4],
    /// The searches run per level: the three pairs' (with the pattern table), else one over
    /// all pairs.
    pairs: Vec<usize>,
    level: Level1,
    /// Candidates made so far (including those dropped as hopeless).
    made_cands: usize,
    /// Wall time spent making candidates (see [`make_candidates`]).
    cand_time: Duration,
    /// Reductions stored so far (including those dropped since).
    made_reds: usize,
    cands: Vec<Candidate>,
    reds: Vec<Reduction>,
    red_index: HashMap<(usize, Cube3), usize>,
    /// See [`Options::order_jobs`].
    order_jobs: bool,
    /// Per 4-move prefix of the phase-1 paths, (kept candidates, phase-1 nodes) at the
    /// current length so far and at the previous length (see [`order_by_yield`]).
    yields: HashMap<[u8; 4], (u64, u64)>,
    prev_yields: HashMap<[u8; 4], (u64, u64)>,
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
        let mut d0 = [0u8; 4];
        let pairs: Vec<usize> = if self.t.p1.pat.is_some() {
            vec![0, 1, 2]
        } else {
            vec![crate::phase1::ALL_PAIRS]
        };
        for &p in &pairs {
            d0[p] = self.t.p1.distance_pair(&root, p);
        }
        let len0 = pairs.iter().map(|&p| d0[p]).min().unwrap();
        let mut run = Run {
            t: self.t,
            cube: *cube,
            threads,
            stop: &stop,
            start,
            limit: hard.map_or(deadline, |h| deadline.min(start + h)),
            p1_cut: None,
            p2_cut: None,
            split2: P2_SPLIT_SLACK,
            have: &have,
            best: None,
            report,
            target: self.opts.target_length,
            stats: Stats::default(),
            root,
            d0,
            pairs,
            level: Level1 {
                len: len0,
                jobs: vec![],
                next: 0,
            },
            made_cands: 0,
            cand_time: Duration::ZERO,
            made_reds: 0,
            cands: vec![],
            reds: vec![],
            red_index: HashMap::new(),
            order_jobs: self.opts.order_jobs,
            yields: HashMap::new(),
            prev_yields: HashMap::new(),
        };
        run.level.jobs = run.level_jobs(len0, false);
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
            run.debug_level("at the end", &run.yields, run.level.len);
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
                // Per (L1, L1 + h2): candidates whose phase 2 has searched every level up to
                // the cap (best - DEEPEST_MARGIN), of all.
                let cap = run.useful_l12();
                let mut hd = std::collections::BTreeMap::new();
                for c in &run.cands {
                    let l1 = c.moves1.len();
                    let e = hd
                        .entry((l1, l1 + c.h2 as usize))
                        .or_insert((0usize, 0usize));
                    e.0 += (l1 + c.done2 as usize > cap) as usize;
                    e.1 += 1;
                }
                eprintln!("searched to the cap {cap} (done, of): {hd:?}");
                if let Some(b) = &run.best {
                    eprintln!("best: {} moves, phases {:?}", b.len(), b.phases);
                }
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
        let (endpoints, reductions) = (endpoints.max(1), reductions.max(1));
        let (mut want1, mut want2) = (endpoints, reductions);
        // Time spent producing reductions (phases 1 and 2) and checking them (phase 3).
        let (mut produce, mut check) = (Duration::ZERO, Duration::ZERO);
        loop {
            if self.stopped() {
                return;
            }
            self.stats.rounds += 1;
            let t0 = Instant::now();
            // Once a solution exists, a round that would run past the time limit is cut short.
            // Each round's phase 1 takes about as long as all the earlier rounds together, and
            // phases 2 and 3 only run after it: without the cut, the last round's endpoints
            // would never be searched (~28% of the run on average). Earlier rounds end long
            // before the cut and are unchanged.
            // Rounds are also kept small (see `ROUND_SHARE`).
            (self.p1_cut, self.p2_cut) = if self.best.is_some() {
                let left = self.limit.saturating_duration_since(t0);
                let small = (t0 - self.start).mul_f64(ROUND_SHARE).max(MIN_ROUND);
                (
                    Some(t0 + left.mul_f64(P1_SHARE).min(small)),
                    Some(t0 + left.mul_f64(P2_SHARE)),
                )
            } else {
                (None, None)
            };
            let tp = Instant::now();
            self.phase1(want1);
            self.stats.time[0] += tp.elapsed();
            debug(
                self.start,
                &format!(
                    "phase 1: {} endpoints (length {}), nodes {}, {:.3} s, candidates made in {:.3} s",
                    self.made_cands,
                    self.level.len,
                    self.stats.nodes1,
                    self.stats.time[0].as_secs_f64(),
                    self.cand_time.as_secs_f64()
                ),
            );
            // A round whose phase 1 was cut short asks phase 2 for proportionally fewer
            // reductions (the usual ratio of reductions to endpoints).
            let want2_now = if self.made_cands < want1 {
                let ratio = reductions as f64 / endpoints as f64;
                ((self.made_cands as f64 * ratio) as usize).clamp(1, want2)
            } else {
                want2
            };
            let tp = Instant::now();
            self.phase2(want2_now);
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
            // Until a first solution exists, phase 3 gets three times that: otherwise a fast
            // phase 1 grows the pool faster than phase 3 can finish the levels that hold the
            // first solution, and the levels only get costlier.
            let factor = if self.best.is_some() { 1.5 } else { 4.5 };
            let budget =
                (produce.mul_f64(factor).saturating_sub(check)).max(Duration::from_millis(10));
            self.phase3_levels(t1 + budget);
            self.stats.time[2] += t1.elapsed();
            check += t1.elapsed();
            if self.stopped() || self.level.len > 40 {
                return;
            }
            want1 = want1.saturating_mul(2);
            want2 = want2.saturating_mul(2);
        }
    }

    /// The jobs of phase-1 length `len`: move prefixes (3 moves at lengths >= 7, or 4 if
    /// `fine` and the length is >= 8, for the order by yield: see [`order_by_yield`]).
    fn level_jobs(&self, len: u8, fine: bool) -> Vec<(u8, Vec<u8>)> {
        let depth = if len < 3 {
            0
        } else if len < 7 {
            2
        } else if fine && len >= 8 {
            4
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
        // Per search, keep only prefixes that can still reach the goal in `len` moves.
        let t = &self.t.p1;
        let mut out = vec![];
        for &pair in &self.pairs {
            if self.d0[pair] > len {
                continue;
            }
            out.extend(
                jobs.iter()
                    .filter(|p| {
                        let mut st = self.root;
                        let mut d = self.d0[pair];
                        for (i, &m) in p.iter().enumerate() {
                            st = st.apply(m as usize, t);
                            d = Mod3Table::decode(d, t.value_pair(&st, pair));
                            if d as usize > len as usize - i - 1 {
                                return false;
                            }
                        }
                        true
                    })
                    .map(|p| (pair as u8, p.clone())),
            );
        }
        out
    }

    /// With EGGG_DEBUG: a phase-1 level's nodes and kept candidates (from its yields).
    fn debug_level(&self, what: &str, yields: &HashMap<[u8; 4], (u64, u64)>, len: u8) {
        if std::env::var_os("EGGG_DEBUG").is_none() {
            return;
        }
        let (kept, nodes) = yields
            .values()
            .fold((0u64, 0u64), |a, v| (a.0 + v.0, a.1 + v.1));
        debug(
            self.start,
            &format!(
                "level {len} {what}: nodes {nodes}, kept {kept} (limit {}), jobs {}/{}",
                self.useful_l12(),
                self.level.next,
                self.level.jobs.len()
            ),
        );
    }

    /// Collect phase-1 endpoints until there are `want` candidates in total (or until the
    /// round's phase-1 cut).
    fn phase1(&mut self, want: usize) {
        while self.made_cands < want
            && !self.stopped()
            && self.level.len <= 40
            && !self.p1_cut.is_some_and(|c| Instant::now() >= c)
        {
            if self.level.next >= self.level.jobs.len() {
                // The level is complete: its yields rank the next level's jobs, once a
                // solution exists (before, every candidate counts as kept, and the order
                // made some first solutions later on the M2).
                let len = self.level.len + 1;
                self.prev_yields = std::mem::take(&mut self.yields);
                self.debug_level("complete", &self.prev_yields, self.level.len);
                let order = self.order_jobs && self.best.is_some();
                let mut jobs = self.level_jobs(len, order);
                if order {
                    order_by_yield(&mut jobs, &self.prev_yields);
                }
                self.level = Level1 { len, jobs, next: 0 };
                continue;
            }
            let need = want - self.made_cands;
            let t = &self.t.p1;
            let (root, d0, len, stop) = (self.root, self.d0, self.level.len, self.stop);
            let jobs = &self.level.jobs;
            let first = self.level.next;
            let threads = self.threads;
            let claimed = AtomicUsize::new(first);
            let found = AtomicUsize::new(0);
            let nodes = AtomicU64::new(0);
            // Raised when enough endpoints are found, when a quarter of the threads have run
            // out of jobs, or at the round's phase-1 cut: running jobs then hand back their
            // unstarted subtrees, so that no thread waits long for the slowest job.
            let close = Close::new(self.p1_cut);
            let idle = AtomicUsize::new(0);
            let out = Mutex::new(vec![]);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let mut local = vec![];
                        let mut n = 0;
                        loop {
                            if stop.load(Ordering::Relaxed) || close.is_raised() {
                                break;
                            }
                            let j = claimed.fetch_add(1, Ordering::Relaxed);
                            if j >= jobs.len() {
                                if (idle.fetch_add(1, Ordering::Relaxed) + 1) * 4 >= threads {
                                    close.raise();
                                }
                                break;
                            }
                            let (pair, ref prefix) = jobs[j];
                            let pair = pair as usize;
                            let (st, d) = root.after_pair(prefix, d0[pair], t, pair);
                            let mut eps = vec![];
                            let mut spill = vec![];
                            let mut per4 = vec![];
                            let mut srch = Search1::new(t, |p: &[u8], f| {
                                eps.push((p.to_vec(), f));
                                true
                            });
                            srch.abort = Some(stop);
                            srch.pair = pair;
                            srch.path.extend_from_slice(prefix);
                            let last = prefix.last().map(|&m| m as usize).unwrap_or(N_MOVES);
                            let left = len - prefix.len() as u8;
                            p1_run(
                                &mut srch, &st, d, left, last, 2, &close, &mut spill, &mut per4,
                            );
                            let job_nodes = srch.nodes;
                            drop(srch);
                            n += job_nodes;
                            // A job of 4 moves or more counts all its nodes for its 4-move
                            // prefix (shorter ones count their subtrees', in `p1_run`).
                            if let Some(k) = prefix4(prefix) {
                                per4.push((k, job_nodes));
                            }
                            let k: usize = eps.iter().map(|e| e.1.count_ones() as usize).sum();
                            if found.fetch_add(k, Ordering::Relaxed) + k >= need {
                                close.raise();
                            }
                            local.push((j, eps, spill, per4));
                        }
                        nodes.fetch_add(n, Ordering::Relaxed);
                        out.lock().unwrap().extend(local);
                    });
                }
            });
            self.stats.nodes1 += nodes.load(Ordering::Relaxed);
            if self.stopped() {
                // The run is over: these endpoints would never be searched (making their
                // candidates took up to seconds past the time limit in long runs).
                break;
            }
            let next = claimed.load(Ordering::Relaxed).min(jobs.len());
            let mut results = out.into_inner().unwrap();
            results.sort_by_key(|r| r.0);
            // Subtrees handed back go first, then the jobs not yet claimed.
            let mut rest: Vec<(u8, Vec<u8>)> = vec![];
            for &(k, n) in results.iter().flat_map(|r| &r.3) {
                self.yields.entry(k).or_default().1 += n;
            }
            for r in &mut results {
                let pair = jobs[r.0].0;
                rest.extend(r.2.drain(..).map(|p| (pair, p)));
            }
            if rest.is_empty() {
                self.level.next = next;
            } else {
                rest.extend(self.level.jobs.drain(next..));
                self.level.jobs = rest;
                self.level.next = 0;
            }
            // New endpoints, in job order; their candidates are made in parallel. Once a
            // solution exists, a candidate whose phase-2 bound already puts its reductions past
            // the deepest level phase 2 will ever search is never needed (and not made).
            // (Levels, jobs and handed-back subtrees are disjoint: no endpoint comes twice.)
            let tc = Instant::now();
            let eps: Vec<(Vec<u8>, u8)> = results.into_iter().flat_map(|r| r.1).collect();
            let limit = self.useful_l12();
            let made = Mutex::new(vec![]);
            let (p2, cube) = (&self.t.p2, &self.cube);
            crate::par::for_chunks(eps.len(), 32, |r| {
                let mut out = vec![];
                make_candidates(p2, cube, &eps[r.clone()], limit, &mut out);
                made.lock().unwrap().push((r.start, out));
            });
            let mut made = made.into_inner().unwrap();
            made.sort_unstable_by_key(|m| m.0);
            self.made_cands += eps.iter().map(|e| e.1.count_ones() as usize).sum::<usize>();
            for c in made.iter().flat_map(|m| &m.1) {
                if let Some(k) = prefix4(&c.moves1) {
                    self.yields.entry(k).or_default().0 += 1;
                }
            }
            self.cands.extend(made.into_iter().flat_map(|m| m.1));
            self.cand_time += tc.elapsed();
        }
        self.stats.endpoints = self.made_cands;
    }

    /// The longest reduction that could still lead to a shorter solution at the deepest
    /// phase-2 level (`DEEPEST_MARGIN`); unlimited before the first solution.
    fn useful_l12(&self) -> usize {
        self.best_len().saturating_sub(DEEPEST_MARGIN)
    }

    /// Drop the candidates and reductions that can no longer lead to a shorter solution
    /// (they only cost memory and scans in long runs).
    fn compact(&mut self) {
        if self.best.is_none() {
            return;
        }
        let limit = self.useful_l12();
        self.cands
            .retain(|c| c.moves1.len() + c.h2.max(c.done2) as usize <= limit);
        let best = self.best_len();
        self.reds
            .retain(|r| r.moves12.len() + (r.h3.max(r.done3) as usize) < best);
        self.red_index.clear();
        for (i, r) in self.reds.iter().enumerate() {
            self.red_index.insert((r.frame, r.c3), i);
        }
    }

    /// Phase 2, level by level on the reduction length, until `want` reductions in total.
    fn phase2(&mut self, want: usize) {
        self.compact();
        // Reductions are produced up to L12 = best - margin, i.e. while a phase 3 of
        // margin - 1 moves could still beat the best solution. Each deeper level has ~16x the
        // reductions of the one above and each is ~13x less likely to have a short enough
        // phase 3, so every level adds about as many chances, for ~16x the phase-2 work: the
        // margin goes from 15 to 13 as the run gets longer (phases 2 and 3 are then cheap
        // next to phase 1). Measured on 20 positions x 30 s: 39.3 -> 39.0 moves.
        let elapsed = self.start.elapsed().as_secs_f64();
        let margin = if elapsed < 3.0 {
            15
        } else if elapsed < 6.0 {
            14
        } else {
            DEEPEST_MARGIN
        };
        let cap = self.best_len().saturating_sub(margin).min(64);
        let mut t12 = self
            .cands
            .iter()
            .map(|c| c.moves1.len() + c.h2.max(c.done2) as usize)
            .min()
            .unwrap_or(99);
        while self.made_reds < want
            && !self.stopped()
            && t12 <= cap
            && !self.p2_cut.is_some_and(|c| Instant::now() >= c)
        {
            for r in self.phase2_level(t12, want - self.made_reds) {
                let key = (r.frame, r.c3);
                match self.red_index.get(&key) {
                    Some(&i) if self.reds[i].moves12.len() <= r.moves12.len() => {}
                    Some(&i) => self.reds[i] = r,
                    None => {
                        self.red_index.insert(key, self.reds.len());
                        self.reds.push(r);
                        self.made_reds += 1;
                    }
                }
            }
            t12 += 1;
        }
        self.stats.reductions = self.reds.len();
    }

    /// Phase-2 searches of total length `t12` for every candidate that can reach it.
    /// Stops starting searches once `need` reductions are found: each level has ~8x the
    /// reductions of the one below, and finishing it would overshoot the round's target (the
    /// candidates not searched keep the level for the next round). Searches still running at
    /// the round's phase-2 cut stop there (keeping the reductions found so far, and the level).
    ///
    /// Once a solution exists, a search with a large slack (depth minus bound, see
    /// `P2_SPLIT_SLACK`) is split when it is started into one task per subtree one or two moves
    /// below its root, which all threads take before starting other searches: one search of
    /// the deepest level can take seconds while the other threads wait. (Before that, a search
    /// keeps only its first solutions, so it stays whole.) The reductions, and their order, are
    /// those of the whole searches.
    fn phase2_level(&mut self, t12: usize, need: usize) -> Vec<Reduction> {
        let t = self.t;
        let best = self.best_len();
        // Before a first solution phase 2 is not capped, so each search keeps at most 64
        // solutions; after it, the cap bounds the searches and all their solutions count.
        let max_sols = if self.best.is_some() { usize::MAX } else { 64 };
        let split = if self.best.is_some() {
            self.split2
        } else {
            u8::MAX
        };
        let cands = &self.cands;
        let todo: Vec<usize> = (0..cands.len())
            .filter(|&i| {
                let c = &cands[i];
                let l1 = c.moves1.len();
                t12 >= l1 + c.h2 as usize && (t12 - l1) as u8 >= c.done2
            })
            .collect();
        let out = Mutex::new(vec![]);
        let nodes = AtomicU64::new(0);
        let stop = self.stop;
        let made = AtomicUsize::new(0);
        let cut = self.p2_cut;
        let past_cut = || cut.is_some_and(|c| Instant::now() >= c);
        // Per search: its jobs (1 if whole), and those that ran to the end.
        let parts: Vec<AtomicUsize> = (0..todo.len()).map(|_| AtomicUsize::new(1)).collect();
        let finished: Vec<AtomicUsize> = (0..todo.len()).map(|_| AtomicUsize::new(0)).collect();
        // Tasks of split searches not yet taken: (search, task number, task).
        let pending: Mutex<VecDeque<(usize, usize, Task2)>> = Mutex::new(VecDeque::new());
        // Threads that may be about to add tasks.
        let splitting = AtomicUsize::new(0);
        let next = AtomicUsize::new(0);
        // Search `todo[k]` whole, or one task of it (number `i`).
        let run = |k: usize, i: usize, task: Option<&Task2>| {
            let c = &cands[todo[k]];
            let (rot, last1, first) = phase2_root(c);
            let rot_inv = transpose(&rot);
            let mut srch = Search2::new(&t.p2);
            srch.abort = Some(stop);
            srch.deadline = cut;
            // Each solution becomes a reduction as soon as it is found, and only the ones that
            // could still beat the best are kept (a deep level can have millions of solutions
            // per search, nearly all hopeless).
            let mut local = vec![];
            let mut seen = 0usize;
            let mut keep = |p: &[u8]| {
                seen += 1;
                let mut x = c.cube;
                for &m in p {
                    x = x.apply_perm(&MOVE_PERMS[m as usize]);
                }
                if let Some(c3) = Cube3::from_reduced(&x) {
                    let h3 = t.p3.lower_bound(&c3);
                    // A reduction whose phase 3 cannot beat the best solution never will
                    // (the best only gets shorter).
                    if c.moves1.len() + p.len() + (h3 as usize) < best {
                        let mut moves12 = c.moves1.clone();
                        moves12.extend(p.iter().map(|&m| rotate_move(m as usize, &rot_inv) as u8));
                        local.push(Reduction {
                            moves12,
                            len1: c.moves1.len(),
                            frame: c.frame,
                            c3,
                            h3,
                            done3: 0,
                        });
                    }
                }
                seen < max_sols
            };
            let l2 = (t12 - c.moves1.len()) as u8;
            match task {
                Some(task) => srch.run_task(task, &mut keep),
                None if t.p2.joint.is_some() => {
                    srch.dfs_joint(c.centre, &c.tau, c.dt, c.df, l2, last1, first, &mut keep)
                }
                None => srch.dfs(c.centre, &c.tau, c.dt, l2, last1, first, &mut keep),
            };
            nodes.fetch_add(srch.nodes, Ordering::Relaxed);
            made.fetch_add(local.len(), Ordering::Relaxed);
            if !srch.aborted {
                finished[k].fetch_add(1, Ordering::Relaxed);
            }
            out.lock().unwrap().push(((k, i), local));
        };
        std::thread::scope(|s| {
            for _ in 0..self.threads.min(todo.len()).max(1) {
                s.spawn(|| {
                    while !stop.load(Ordering::Relaxed) {
                        // Tasks of started searches first.
                        let job = pending.lock().unwrap().pop_front();
                        if let Some((k, i, task)) = job {
                            if !past_cut() {
                                run(k, i, Some(&task));
                            }
                            continue;
                        }
                        splitting.fetch_add(1, Ordering::SeqCst);
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= todo.len() {
                            splitting.fetch_sub(1, Ordering::SeqCst);
                            if splitting.load(Ordering::SeqCst) == 0
                                && pending.lock().unwrap().is_empty()
                            {
                                break;
                            }
                            std::thread::yield_now();
                            continue;
                        }
                        if made.load(Ordering::Relaxed) >= need || past_cut() {
                            splitting.fetch_sub(1, Ordering::SeqCst);
                            continue;
                        }
                        let c = &cands[todo[k]];
                        let l2 = (t12 - c.moves1.len()) as u8;
                        let slack = l2.saturating_sub(c.h2);
                        if slack < split {
                            splitting.fetch_sub(1, Ordering::SeqCst);
                            run(k, 0, None);
                            continue;
                        }
                        let (_, last1, first) = phase2_root(c);
                        let depth = if slack > split { 2 } else { 1 };
                        let tasks = Search2::new(&t.p2)
                            .tasks(c.centre, &c.tau, c.dt, c.df, l2, last1, first, depth);
                        parts[k].store(tasks.len(), Ordering::Relaxed);
                        pending
                            .lock()
                            .unwrap()
                            .extend(tasks.into_iter().enumerate().map(|(i, x)| (k, i, x)));
                        splitting.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        for (k, &i) in todo.iter().enumerate() {
            if finished[k].load(Ordering::Relaxed) == parts[k].load(Ordering::Relaxed) {
                let c = &mut self.cands[i];
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
                None => srch.search_depth(&r.c3, d),
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

/// Load or build the phase-1 tables in `dir`; `big`: the large staging table (no symmetry
/// reduction, faster lookups) instead of the reduced one.
pub fn load_p1(dir: &std::path::Path, big: bool, threads: usize, verbose: bool) -> P1Tables {
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
    };
    let bytes = table.data.as_bytes_mut();
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
        let bytes = table.data.as_bytes();
        crate::table::save(&p1_path, p1_name, bytes).expect("saving table");
    }
    let mut p1 = P1Tables::with_table(staging, table);
    p1.big = big;
    p1
}

/// Load or build all tables in `dir`.
pub fn load_tables(dir: &std::path::Path, h48_k: u32, threads: usize, verbose: bool) -> Tables {
    load_tables_with(dir, h48_k, false, false, (None, None), threads, verbose)
}

/// Load (or build) the phase-1 pattern table into `p1`: the reduced one (0.9 GB) or the
/// large one (`big`, 41 GB). Phase 1 then searches per colour pair.
pub fn load_pat(
    p1: &mut P1Tables,
    dir: &std::path::Path,
    big: bool,
    threads: usize,
    verbose: bool,
) {
    p1.pat = Some(crate::p1pat::load(dir, &p1.staging, big, threads, verbose));
}

/// Load (or build from the staging table) the phase-1 move-mask table into `p1`: the large
/// one (`big`, 18.9 GB; needs the large staging table) or the reduced one (1.6 GB).
pub fn load_mask(p1: &mut P1Tables, dir: &std::path::Path, big: bool, verbose: bool) {
    p1.mask = Some(crate::p1mask::load(dir, p1, big, verbose));
}

/// Load or build the phase-2 tables in `dir`; `joint`: also the joint table (1.05 GB).
pub fn load_p2(dir: &std::path::Path, joint: bool, threads: usize, verbose: bool) -> P2Tables {
    let p2_path = dir.join("p2-tau.tab");
    let mut p2 = {
        let n = crate::phase2::N_TAU;
        let mut table = Mod3Table {
            data: crate::table::HugeBuf::zeroed(n.div_ceil(32)),
        };
        let bytes = table.data.as_bytes_mut();
        if crate::table::load(&p2_path, "p2-tau", bytes).is_ok() {
            P2Tables::build(Some(table), threads)
        } else {
            if verbose {
                eprintln!("generating phase-2 pairing table...");
            }
            let t = P2Tables::build(None, threads);
            let bytes = t.tau_table.data.as_bytes();
            crate::table::save(&p2_path, "p2-tau", bytes).expect("saving table");
            t
        }
    };
    if joint {
        let mut j = crate::p2joint::Joint::new(&p2);
        let path = dir.join("p2-joint2.tab");
        j.table = crate::table::HugeBuf::zeroed(j.bytes() / 8);
        let bytes = j.table.as_bytes_mut();
        if crate::table::load(&path, "p2-joint2", bytes).is_err() {
            if verbose {
                eprintln!(
                    "generating the joint phase-2 table ({:.1} GB)...",
                    j.bytes() as f64 / 1e9
                );
            }
            j.generate(&p2, threads, verbose);
            let bytes = j.table.as_bytes();
            crate::table::save(&path, "p2-joint2", bytes).expect("saving table");
        }
        p2.joint = Some(j);
    }
    p2
}

/// Load or build all tables in `dir`; `joint`: also the joint phase-2 table (1.05 GB);
/// `p1_big`: the large phase-1 staging table (1.2 GB); `p1_extra`: the phase-1 pattern table
/// and move-mask table (each `Some(big)`: reduced or large; the large masks need `p1_big`).
pub fn load_tables_with(
    dir: &std::path::Path,
    h48_k: u32,
    joint: bool,
    p1_big: bool,
    p1_extra: (Option<bool>, Option<bool>),
    threads: usize,
    verbose: bool,
) -> Tables {
    let t0 = Instant::now();
    // The phase-2 and phase-3 tables load in their own threads meanwhile (loads overlap;
    // missing tables are still generated one at a time: see `LoadGroup`).
    let p2 = {
        let dir = dir.to_path_buf();
        std::thread::spawn(move || {
            let _g = crate::table::LoadGroup::enter();
            load_p2(&dir, joint, threads, verbose)
        })
    };
    let p3 = {
        let dir = dir.to_path_buf();
        std::thread::spawn(move || {
            let _g = crate::table::LoadGroup::enter();
            P3Tables::load_or_build(&dir, h48_k, threads, verbose)
        })
    };
    let g = crate::table::LoadGroup::enter();
    let mut p1 = load_p1(dir, p1_big, threads, verbose);
    if let Some(big) = p1_extra.0 {
        load_pat(&mut p1, dir, big, threads, verbose);
    }
    if let Some(big) = p1_extra.1 {
        assert!(
            p1_big || !big || p1.pat.is_some(),
            "the large move-mask table needs the large staging table"
        );
        load_mask(&mut p1, dir, big, verbose);
    }
    drop(g);
    let p2 = p2.join().expect("loading the phase-2 tables");
    let p3 = p3.join().expect("loading the phase-3 tables");
    // Huge pages for the big tables, in parallel.
    std::thread::scope(|s| {
        s.spawn(|| p1.table.data.collapse());
        if let Some(pt) = &p1.pat {
            s.spawn(|| pt.table.data.collapse());
        }
        if let Some(m) = &p1.mask {
            s.spawn(|| m.collapse());
        }
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
    use crate::p2joint::ORDER;
    use rand::SeedableRng;
    use std::collections::HashSet;

    /// The candidate of one endpoint frame (see [`make_candidates`]).
    fn candidate(
        t: &Tables,
        cube: &Cube,
        moves1: &[u8],
        frame: usize,
        limit: usize,
    ) -> Option<Candidate> {
        let mut out = vec![];
        make_candidates(
            &t.p2,
            cube,
            &[(moves1.to_vec(), 1 << frame)],
            limit,
            &mut out,
        );
        out.pop()
    }

    /// The former candidate construction: a rotation search per standardisation, the full
    /// joint-table walk, and the limit applied afterwards.
    fn candidate_ref(
        t: &Tables,
        cube: &Cube,
        moves1: &[u8],
        frame: usize,
        limit: usize,
    ) -> Option<Candidate> {
        let mut c = *cube;
        for &m in moves1 {
            c = c.apply_perm(&MOVE_PERMS[m as usize]);
        }
        let x = crate::phase2::tests::standardise_ref(&c, frame);
        let centre = t.p2.centre_coord_of(&x);
        let cd = t.p2.centre_dist[centre as usize];
        if cd == crate::table::UNSEEN {
            return None;
        }
        let tau = t.p2.tau_of(&x);
        let (dt, df) = match &t.p2.joint {
            Some(j) => {
                let (ud, tf, pll) = fb_view_ref(&t.p2, &x);
                (
                    j.distance_ref(&t.p2, centre / 1680, &tau, centre & 1),
                    j.distance_ref(&t.p2, ud, &tf, pll),
                )
            }
            None => (t.p2.tau_distance(&tau), 0),
        };
        let c = Candidate {
            moves1: moves1.to_vec(),
            frame,
            cube: x,
            centre,
            tau,
            dt,
            df,
            h2: cd.max(dt).max(df),
            done2: 0,
        };
        (c.moves1.len() + c.h2 as usize <= limit).then_some(c)
    }

    /// The joint-table state (U/D pattern, tau, PLL bit) of the cube rotated by x (about the
    /// R/L axis): the F/B view of a phase-2 state, from the rotated cube itself.
    fn fb_view_ref(p2: &P2Tables, x: &Cube) -> (u32, [u8; 16], u32) {
        let m = *crate::cube::ROTATIONS
            .iter()
            .find(|m| m[0][0] == 1 && m[1][1] == 0)
            .unwrap();
        let y = crate::moves::Conj::new(m).apply(x);
        let mut mask = 0u8;
        for (k, p) in [0usize, 1, 2, 3, 12, 13, 14, 15].into_iter().enumerate() {
            if y.ct[p] / 4 == crate::cube::U as u8 {
                mask |= 1 << k;
            }
        }
        let pll = crate::phase2::pll_parity(&y, &p2.low);
        (p2.rank70(mask) as u32, p2.tau_of(&y), pll)
    }

    /// Tables with the joint phase-2 table, from $EGGG_TABLES (default ./tables) with the H48
    /// table h$EGGG_TEST_H48 (default h3).
    fn joint_tables() -> Tables {
        let dir = std::env::var_os("EGGG_TABLES")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "tables".into());
        let k = std::env::var("EGGG_TEST_H48")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        load_tables_with(&dir, k, true, false, (None, None), 8, false)
    }

    type Endpoints = Vec<(Vec<u8>, u8)>;

    /// Phase-1 endpoints (with frames) of random cubes: per cube, its endpoints of the
    /// shortest few lengths, up to `per_cube`.
    fn random_endpoints(
        t: &Tables,
        cubes: usize,
        per_cube: usize,
        seed: u64,
    ) -> Vec<(Cube, Endpoints)> {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..cubes)
            .map(|_| {
                let cube = Cube::random(&mut rng);
                let root = P1State::from_cube(&cube);
                let d0 = t.p1.distance(&root);
                let mut eps = vec![];
                for len in d0..d0 + 4 {
                    let mut s = Search1::new(&t.p1, |p: &[u8], f| {
                        eps.push((p.to_vec(), f));
                        eps.len() < per_cube
                    });
                    s.dfs(&root, d0, len, N_MOVES);
                    if eps.len() >= per_cube {
                        break;
                    }
                }
                (cube, eps)
            })
            .collect()
    }

    fn same(a: &Candidate, b: &Candidate) -> bool {
        (
            &a.moves1, a.frame, a.cube, a.centre, a.tau, a.dt, a.df, a.h2, a.done2,
        ) == (
            &b.moves1, b.frame, b.cube, b.centre, b.tau, b.dt, b.df, b.h2, b.done2,
        )
    }

    /// The candidates made (cached standardisation, bounded and interleaved joint-table walks)
    /// are exactly the former construction's, at every limit.
    #[test]
    fn candidates_match_the_former_construction() {
        let t = joint_tables();
        let (mut kept, mut total) = (0usize, 0usize);
        for (cube, eps) in random_endpoints(&t, 12, 400, 11) {
            let Some(l1) = eps.iter().map(|e| e.0.len()).min() else {
                continue;
            };
            for limit in (l1 + 6..l1 + 22).chain([usize::MAX]) {
                let mut new = vec![];
                // Uneven batches, as the parallel chunks give.
                for chunk in eps.chunks(37) {
                    make_candidates(&t.p2, &cube, chunk, limit, &mut new);
                }
                let mut old = vec![];
                for (moves1, frames) in &eps {
                    for frame in 0..3 {
                        if frames >> frame & 1 == 1 {
                            total += 1;
                            old.extend(candidate_ref(&t, &cube, moves1, frame, limit));
                        }
                    }
                }
                kept += old.len();
                assert_eq!(new.len(), old.len(), "limit {limit}");
                for (a, b) in new.iter().zip(&old) {
                    assert!(
                        same(a, b),
                        "limit {limit}: {:?} frame {}",
                        b.moves1,
                        b.frame
                    );
                }
            }
        }
        assert!(kept > 1000 && kept < total, "kept {kept} of {total}");
    }

    /// Joint-table walks of random phase-2 states (deeper than endpoints) match the former
    /// walk, with and without bounds.
    #[test]
    fn joint_walks_match_the_former_walk() {
        use crate::phase2::P2_MOVES;
        use rand::Rng;
        let t = joint_tables();
        let (p2, j) = (&t.p2, t.p2.joint.as_ref().unwrap());
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
        let mut states = vec![];
        let mut want = vec![];
        for _ in 0..3000 {
            let mut c = Cube::SOLVED;
            for _ in 0..rng.gen_range(0..40) {
                c = c.apply_perm(&MOVE_PERMS[P2_MOVES[rng.gen_range(0..17)] as usize]);
            }
            let cc = p2.centre_coord_of(&c);
            let tau = p2.tau_of(&c);
            let d = j.distance_ref(p2, cc / 1680, &tau, cc & 1);
            assert_eq!(j.distance(cc / 1680, &tau, cc & 1), d);
            let max = rng.gen_range(0..=d + 2);
            states.push((cc / 1680, tau, cc & 1, max));
            want.push((d <= max).then_some(d));
        }
        let mut got = vec![];
        j.distances_at_most(&states, &mut got);
        assert_eq!(got, want);
    }

    /// The F/B view of the joint table: its state is the rotated cube's, its distance is the
    /// same whether walked in the rotated frame or over our own moves (the rotated move set
    /// is the phase-2 move set up to symmetries of the table), it is admissible (at most the
    /// length of the move sequence that made the state), and the view's tau follows the moves.
    #[test]
    fn fb_view_distances() {
        use crate::phase2::P2_MOVES;
        use rand::Rng;
        let t = joint_tables();
        let (p2, j) = (&t.p2, t.p2.joint.as_ref().unwrap());
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(4);
        // The former walk over our own moves, reading the view of each child.
        let walk_ours = |cc: u32, tau: &[u8; 16]| {
            let get = |cc: u32, tau: &[u8; 16]| {
                let (ud, tf) = j.fb.map(cc, tau);
                j.get(j.index(ud, &tf, cc & 1))
            };
            let (mut cc, mut tau, mut d) = (cc, *tau, 0u8);
            loop {
                let v = get(cc, &tau);
                let next = (0..17)
                    .map(|i| (p2.centre_move[cc as usize][i], p2.tau_move(&tau, i)))
                    .find(|(c2, t2)| get(*c2, t2) == (v + 2) % 3);
                match next {
                    Some((c2, t2)) if d < 40 => (cc, tau, d) = (c2, t2, d + 1),
                    _ => return d,
                }
            }
        };
        let (mut sum_ud, mut sum_fb, mut sum_max, mut n) = (0usize, 0usize, 0usize, 0usize);
        for _ in 0..3000 {
            let k = rng.gen_range(0..30);
            let mut c = Cube::SOLVED;
            let (mut cc, mut tau) = (p2.centre_coord_of(&c), p2.tau_of(&c));
            let mut tf = j.fb.tau(&tau);
            for _ in 0..k {
                let i = rng.gen_range(0..17);
                c = c.apply_perm(&MOVE_PERMS[P2_MOVES[i] as usize]);
                (cc, tau) = (p2.centre_move[cc as usize][i], p2.tau_move(&tau, i));
                tf = j.fb.tau_move(&tf, i);
                assert_eq!(tf[..12], j.fb.tau(&tau)[..12]);
            }
            assert_eq!(cc, p2.centre_coord_of(&c));
            let (ud, tr, pll) = fb_view_ref(p2, &c);
            assert_eq!((ud, &tr[..12], pll), (j.fb.ud(cc), &tf[..12], cc & 1));
            let df = j.distance_fb(cc, &tau);
            assert_eq!(df, j.distance_ref(p2, ud, &tr, pll));
            assert_eq!(df, walk_ours(cc, &tau));
            assert!(df as usize <= k, "F/B view distance {df} after {k} moves");
            let dj = j.distance(cc / 1680, &tau, cc & 1);
            assert!(dj as usize <= k);
            if k >= 20 {
                (sum_ud, sum_fb, sum_max, n) = (
                    sum_ud + dj as usize,
                    sum_fb + df as usize,
                    sum_max + dj.max(df) as usize,
                    n + 1,
                );
            }
        }
        // Each view alone is about as tight; their maximum is tighter.
        let m = |x: usize| x as f64 / n as f64;
        eprintln!(
            "{n} states: U/D {:.2}, F/B {:.2}, max {:.2}",
            m(sum_ud),
            m(sum_fb),
            m(sum_max)
        );
        assert!(sum_max > sum_ud && sum_max > sum_fb);
    }

    /// The joint-table search (both views) finds exactly the solutions of the search with the
    /// pairing table alone, in the same order, at every depth from the pairing bound up:
    /// candidates of random cubes' phase-1 endpoints, all depths up to two past the bound.
    #[test]
    fn joint_search_finds_the_pairing_search_solutions() {
        let t = joint_tables();
        let p2 = &t.p2;
        let (mut sols, mut searches, mut nodes_j, mut nodes_p) = (0usize, 0usize, 0u64, 0u64);
        let mut tighter = 0;
        for (cube, eps) in random_endpoints(&t, 6, 40, 17) {
            let mut cands = vec![];
            make_candidates(p2, &cube, &eps, usize::MAX, &mut cands);
            for c in cands.iter().step_by(3) {
                let (_, last1, first) = phase2_root(c);
                let cd = p2.centre_dist[c.centre as usize];
                let dt = p2.tau_distance(&c.tau);
                let hp = cd.max(dt);
                assert!(hp <= c.h2);
                tighter += (hp < c.h2) as usize;
                // Every depth from the pairing bound to two past the joint one, while the
                // pairing-table searches stay small.
                let mut budget = 3_000_000u64;
                for left in hp..=c.h2 + 2 {
                    let mut want = vec![];
                    let mut s = Search2::new(p2);
                    s.dfs(c.centre, &c.tau, dt, left, last1, first, &mut |p| {
                        want.push(p.to_vec());
                        true
                    });
                    nodes_p += s.nodes;
                    budget = budget.saturating_sub(s.nodes);
                    let mut got = vec![];
                    if left >= c.h2 {
                        let mut s = Search2::new(p2);
                        s.dfs_joint(c.centre, &c.tau, c.dt, c.df, left, last1, first, &mut |p| {
                            got.push(p.to_vec());
                            true
                        });
                        nodes_j += s.nodes;
                    }
                    assert_eq!(got, want, "{:?} frame {}, depth {left}", c.moves1, c.frame);
                    sols += want.len();
                    searches += 1;
                    if budget == 0 {
                        break;
                    }
                }
            }
        }
        eprintln!(
            "{searches} searches, {sols} solutions; nodes: pairing table {nodes_p}, joint {nodes_j}; \
             {tighter} candidates with a tighter bound"
        );
        assert!(
            sols > 500 && tighter > 0,
            "{sols} solutions, {tighter} tighter"
        );
    }

    /// Candidate construction timings (single thread), former and current, on phase-1
    /// endpoints of random cubes at a few limits (L1 of the shortest endpoint + `extra`):
    /// `cargo test --release candidate_timings -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn candidate_timings() {
        let t = joint_tables();
        let (p2, j) = (&t.p2, t.p2.joint.as_ref().unwrap());
        let sets = random_endpoints(&t, 40, 1000, 21);
        let n: usize = sets
            .iter()
            .flat_map(|s| &s.1)
            .map(|e| e.1.count_ones() as usize)
            .sum();
        // Table reads per joint-table walk: all 17 children in index order, the current
        // state re-read at each step (former), and the walk's.
        let (mut walks, mut steps, mut former, mut now) = (0, 0, 0, 0);
        for (cube, eps) in &sets {
            for (moves1, frames) in eps {
                for frame in 0..3 {
                    if frames >> frame & 1 == 1 {
                        let mut c = *cube;
                        for &m in moves1 {
                            c = c.apply_perm(&MOVE_PERMS[m as usize]);
                        }
                        let x = standardise(&c, frame);
                        let cc = p2.centre_coord_of(&x);
                        if p2.centre_dist[cc as usize] != crate::table::UNSEEN {
                            let tau = p2.tau_of(&x);
                            let (s, all) = j.walk_reads(p2, cc / 1680, &tau, cc & 1, None);
                            let (_, w) = j.walk_reads(p2, cc / 1680, &tau, cc & 1, Some(&ORDER));
                            (walks, steps, former, now) =
                                (walks + 1, steps + s, former + all + s + 1, now + w + 1);
                        }
                    }
                }
            }
        }
        let w = walks as f64;
        eprintln!(
            "{n} candidates, {walks} walks: {:.2} steps, table reads {:.1} (former) / {:.1}",
            steps as f64 / w,
            former as f64 / w,
            now as f64 / w
        );
        for extra in [usize::MAX, 14, 12, 11] {
            let lim = |eps: &[(Vec<u8>, u8)]| {
                let l1 = eps.iter().map(|e| e.0.len()).min().unwrap_or(0);
                l1.saturating_add(extra)
            };
            let t0 = Instant::now();
            let mut kept = 0;
            for (cube, eps) in &sets {
                for (moves1, frames) in eps {
                    for f in 0..3 {
                        if frames >> f & 1 == 1 {
                            kept += candidate_ref(&t, cube, moves1, f, lim(eps)).is_some() as usize;
                        }
                    }
                }
            }
            let old = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
            let t0 = Instant::now();
            let mut out = vec![];
            for (cube, eps) in &sets {
                for chunk in eps.chunks(32) {
                    make_candidates(p2, cube, chunk, lim(eps), &mut out);
                }
            }
            let new = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
            assert_eq!(out.len(), kept);
            eprintln!(
                "L1 + {extra:>2}: {kept:5} of {n} kept; us per candidate: former {old:.2}, now {new:.2}"
            );
        }
    }

    /// Opening phase 2 with moves of phase 1's trailing axis, and phase 1 not reporting the
    /// endpoints that this covers, gives exactly the old phase 1+2 sequences, each once.
    #[test]
    fn phase_boundary_is_one_to_one() {
        use crate::phase1::trailing_run;
        use crate::phase2::first_moves;
        use rand::Rng;
        let t = load_tables(std::path::Path::new("tables"), 3, 8, false);
        // All endpoints of length `len` (with frames) under a frame rule, by a plain DFS.
        fn endpoints(
            t: &Tables,
            s: &P1State,
            d: u8,
            left: u8,
            path: &mut Vec<u8>,
            new_rule: bool,
            out: &mut Vec<(Vec<u8>, u8)>,
        ) {
            if left == 0 {
                if d == 0 && !path.is_empty() {
                    let run = if new_rule {
                        trailing_run(path)
                    } else {
                        1 << path[path.len() - 1]
                    };
                    let f = t.p1.endpoint_frames(s, run);
                    if f != 0 {
                        out.push((path.clone(), f));
                    }
                }
                return;
            }
            let last = path.last().map(|&m| m as usize).unwrap_or(N_MOVES);
            let mut allowed = ALLOWED_AFTER[last];
            while allowed != 0 {
                let m = allowed.trailing_zeros() as usize;
                allowed &= allowed - 1;
                let c = s.apply(m, &t.p1);
                let dc = Mod3Table::decode(d, t.p1.value(&c));
                if dc < left {
                    path.push(m as u8);
                    endpoints(t, &c, dc, left - 1, path, new_rule, out);
                    path.pop();
                }
            }
        }
        // Canonical form: each run of same-axis moves sorted by face (they commute).
        fn canonical(mut v: Vec<u8>) -> Vec<u8> {
            let mut i = 0;
            while i < v.len() {
                let axis = (v[i] / 3) % 3;
                let mut j = i;
                while j < v.len() && (v[j] / 3) % 3 == axis {
                    j += 1;
                }
                v[i..j].sort_by_key(|&m| m / 3);
                i = j;
            }
            v
        }
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
        let mut total = 0;
        for _ in 0..10 {
            let mut cube = Cube::SOLVED;
            for _ in 0..rng.gen_range(4..7) {
                cube = cube.apply_perm(&MOVE_PERMS[rng.gen_range(0..N_MOVES)]);
            }
            let root = P1State::from_cube(&cube);
            let d0 = t.p1.distance(&root);
            if d0 < 2 {
                continue;
            }
            let k = d0 as usize + 3;
            let mut seqs: [Vec<(usize, Vec<u8>)>; 2] = [vec![], vec![]];
            for (scheme, seq) in seqs.iter_mut().enumerate() {
                let new_rule = scheme == 1;
                for l1 in d0.max(1) as usize..=k {
                    let mut eps = vec![];
                    endpoints(&t, &root, d0, l1 as u8, &mut vec![], new_rule, &mut eps);
                    if new_rule {
                        // The real search reports the same endpoints.
                        let mut got = vec![];
                        let mut srch = Search1::new(&t.p1, |p: &[u8], f| {
                            got.push((p.to_vec(), f));
                            true
                        });
                        srch.dfs(&root, d0, l1 as u8, N_MOVES);
                        drop(srch);
                        let mut want = eps.clone();
                        want.sort();
                        got.sort();
                        assert_eq!(got, want);
                    }
                    for (moves1, frames) in eps {
                        for frame in 0..3 {
                            if frames >> frame & 1 == 0 {
                                continue;
                            }
                            let Some(c) = candidate(&t, &cube, &moves1, frame, usize::MAX) else {
                                continue;
                            };
                            let rot = frame_rotation(frame);
                            let rot_inv = transpose(&rot);
                            let rotated: Vec<u8> = moves1
                                .iter()
                                .map(|&m| rotate_move(m as usize, &rot) as u8)
                                .collect();
                            let last1 = *rotated.last().unwrap() as usize;
                            let first = if new_rule {
                                first_moves(&rotated)
                            } else {
                                ALLOWED_AFTER[last1]
                            };
                            for l2 in 0..=(k - l1) as u8 {
                                let mut srch = Search2::new(&t.p2);
                                srch.dfs(c.centre, &c.tau, c.dt, l2, last1, first, &mut |p| {
                                    let mut v = moves1.clone();
                                    v.extend(
                                        p.iter().map(|&m| rotate_move(m as usize, &rot_inv) as u8),
                                    );
                                    seq.push((frame, canonical(v)));
                                    true
                                });
                            }
                        }
                    }
                }
            }
            let [old, new] = seqs;
            let mut dedup = new.clone();
            dedup.sort();
            dedup.dedup();
            assert_eq!(dedup.len(), new.len(), "the new scheme repeats a reduction");
            let mut old = old;
            old.sort();
            let n = old.len();
            old.dedup();
            assert_eq!(old.len(), n, "the old scheme repeats a reduction");
            // Every old reduction is a new one, or a new one followed by outer (3x3) moves
            // that phase 3 makes anyway (old reductions whose trailing run held a phase-2
            // outer move with no phase-2 moves after it).
            let newset: HashSet<(usize, Vec<u8>)> = dedup.iter().cloned().collect();
            let oldset: HashSet<(usize, Vec<u8>)> = old.iter().cloned().collect();
            let extra: Vec<_> = newset.difference(&oldset).take(8).collect();
            assert!(
                extra.is_empty(),
                "{} new-only reductions, e.g. {extra:?}",
                newset.difference(&oldset).count()
            );
            // Shorter sequences with outer moves of the final run(s) moved to the end and
            // dropped (up to `n` of them).
            fn stripped(v: &[u8], n: usize, out: &mut Vec<Vec<u8>>) {
                if n == 0 || v.is_empty() {
                    return;
                }
                let axis = (v[v.len() - 1] / 3) % 3;
                let start = v
                    .iter()
                    .rposition(|&m| (m / 3) % 3 != axis)
                    .map_or(0, |i| i + 1);
                for i in start..v.len() {
                    if v[i] < 18 {
                        let mut w = v.to_vec();
                        w.remove(i);
                        out.push(w.clone());
                        stripped(&w, n - 1, out);
                    }
                }
            }
            let mut lost = 0;
            for (frame, v) in old.iter().filter(|x| !newset.contains(*x)) {
                let mut ws = vec![];
                stripped(v, 3, &mut ws);
                assert!(
                    ws.into_iter()
                        .any(|w| newset.contains(&(*frame, canonical(w)))),
                    "old reduction {v:?} (frame {frame}) is not covered"
                );
                lost += 1;
            }
            assert!(lost < old.len());
            total += old.len();
        }
        assert!(total > 100, "too few reductions to test ({total})");
    }

    /// A phase-2 search past its deadline gives up within ~4096 nodes and says so; one with a
    /// later deadline finds the same solutions as one without.
    #[test]
    fn phase2_deadline() {
        use crate::phase2::P2_MOVES;
        let t = load_tables(std::path::Path::new("tables"), 3, 8, false);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(11);
        let search = |c: &Candidate, left: u8, deadline: Option<Instant>| {
            let mut srch = Search2::new(&t.p2);
            srch.deadline = deadline;
            let mut sols = vec![];
            let all = ALLOWED_AFTER[N_MOVES];
            srch.dfs(c.centre, &c.tau, c.dt, left, N_MOVES, all, &mut |p| {
                sols.push(p.to_vec());
                true
            });
            (sols, srch.aborted, srch.nodes)
        };
        let mut big = 0;
        for _ in 0..6 {
            let mut cube = Cube::SOLVED;
            for _ in 0..12 {
                let m = P2_MOVES[rand::Rng::gen_range(&mut rng, 0..P2_MOVES.len())];
                cube = cube.apply_perm(&MOVE_PERMS[m as usize]);
            }
            let c = candidate(&t, &cube, &[], 0, usize::MAX).unwrap();
            let left = c.h2 + 2;
            let (want, aborted, nodes) = search(&c, left, None);
            assert!(!aborted);
            let later = Instant::now() + Duration::from_secs(3600);
            let (got, aborted, _) = search(&c, left, Some(later));
            assert!(!aborted);
            assert_eq!(got, want);
            if nodes > 10_000 {
                let (_, aborted, n) = search(&c, left, Some(Instant::now()));
                assert!(aborted && n <= 4096, "{n} nodes");
                big += 1;
            }
        }
        assert!(big > 0, "no search big enough to test the deadline");
    }

    /// A phase-2 search split into tasks one or two moves below the root finds exactly the
    /// solutions of the whole search, in the same order (with and without the joint table).
    #[test]
    fn phase2_tasks_find_the_same_solutions() {
        use crate::phase2::P2_MOVES;
        let dir = std::path::Path::new("tables");
        let small = load_tables(dir, 3, 8, false);
        let joint = load_tables_with(dir, 3, true, false, (None, None), 8, false);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(12);
        let mut total = 0;
        for t in [&small, &joint] {
            for _ in 0..8 {
                let mut cube = Cube::SOLVED;
                for _ in 0..14 {
                    let m = P2_MOVES[rand::Rng::gen_range(&mut rng, 0..P2_MOVES.len())];
                    cube = cube.apply_perm(&MOVE_PERMS[m as usize]);
                }
                let c = candidate(t, &cube, &[], 0, usize::MAX).unwrap();
                let (_, last1, first) = phase2_root(&c);
                for left in c.dt..c.h2 + 4 {
                    let mut want = vec![];
                    let mut srch = Search2::new(&t.p2);
                    let mut found = |p: &[u8]| {
                        want.push(p.to_vec());
                        true
                    };
                    if t.p2.joint.is_some() {
                        srch.dfs_joint(
                            c.centre, &c.tau, c.dt, c.df, left, last1, first, &mut found,
                        );
                    } else {
                        srch.dfs(c.centre, &c.tau, c.dt, left, last1, first, &mut found);
                    }
                    for depth in 1..=2 {
                        let tasks =
                            srch.tasks(c.centre, &c.tau, c.dt, c.df, left, last1, first, depth);
                        let mut got = vec![];
                        for task in &tasks {
                            srch.run_task(task, &mut |p| {
                                got.push(p.to_vec());
                                true
                            });
                        }
                        assert_eq!(got, want, "left {left}, depth {depth}");
                    }
                    total += want.len();
                }
            }
        }
        assert!(total > 1000, "too few solutions to test ({total})");
    }

    /// [`order_by_yield`] permutes the jobs, best smoothed yield first (4-move prefixes
    /// towards their 3-move prefix's estimate, those towards the mean; unknown prefixes at
    /// their parent's estimate), ties in their order.
    #[test]
    fn job_order_is_a_permutation_by_yield() {
        let jobs: Vec<(u8, Vec<u8>)> = vec![
            (3, vec![0, 3, 6]),
            (3, vec![0, 3, 7, 1]),
            (3, vec![1, 4]),
            (3, vec![2, 5, 8, 1]),
            (3, vec![0, 3, 6, 9]),
            (3, vec![0, 3, 6, 10, 2]),
            (3, vec![1, 5]),
        ];
        let mut o = jobs.clone();
        order_by_yield(&mut o, &HashMap::new());
        assert_eq!(o, jobs);
        // Mean rate 80 / 4000 = 0.02, mean nodes 4000 / 3 per 3-move prefix, 1000 per
        // 4-move prefix. 3-move estimates: [0, 3, 6] (20 + 0.02 * 4000 / 3) / (2000 + 4000 / 3)
        // = 0.014, [0, 3, 7] 0.03286, [2, 5, 8] 0.01571. 4-move: [0, 3, 7, 1] (50 + 32.86) /
        // 2000 = 0.04143, [2, 5, 8, 1] (unknown) 0.01571, [0, 3, 6, 9] 14 / 2000 = 0.007,
        // [0, 3, 6, 10] 34 / 2000 = 0.017; [1, 4] and [1, 5]: the mean, 0.02.
        let y = HashMap::from([
            ([0, 3, 6, 9], (0, 1000)),
            ([0, 3, 6, 10], (20, 1000)),
            ([0, 3, 7, 1], (50, 1000)),
            ([2, 5, 8, 0], (10, 1000)),
        ]);
        order_by_yield(&mut o, &y);
        let at = |i: usize| jobs[i].clone();
        assert_eq!(o, vec![at(1), at(2), at(6), at(5), at(3), at(0), at(4)]);
    }

    /// With the jobs of each new length ordered by yield, phase 1 finds exactly the
    /// candidates of the plain order at every completed length (jobs cut short and resumed
    /// as the driver does), and the order does change.
    #[test]
    fn ordered_levels_find_the_same_candidates() {
        let t = load_tables(std::path::Path::new("tables"), 3, 8, false);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(17);
        for _ in 0..2 {
            let cube = Cube::random(&mut rng);
            let root = P1State::from_cube(&cube);
            let pairs = vec![crate::phase1::ALL_PAIRS];
            let mut d0 = [0u8; 4];
            d0[pairs[0]] = t.p1.distance_pair(&root, pairs[0]);
            let last = d0[pairs[0]] + 4;
            let mut found = vec![];
            let mut ys = vec![];
            for order_jobs in [false, true] {
                let (stop, have) = (AtomicBool::new(false), AtomicBool::new(false));
                let mut report = |_: &Solution| {};
                let start = Instant::now();
                let mut run = Run {
                    t: &t,
                    cube,
                    threads: 4,
                    stop: &stop,
                    have: &have,
                    start,
                    limit: start + Duration::from_secs(3600),
                    p1_cut: None,
                    p2_cut: None,
                    split2: u8::MAX,
                    best: None,
                    report: &mut report,
                    target: None,
                    stats: Stats::default(),
                    root,
                    d0,
                    pairs: pairs.clone(),
                    level: Level1 {
                        len: d0[crate::phase1::ALL_PAIRS],
                        jobs: vec![],
                        next: 0,
                    },
                    made_cands: 0,
                    cand_time: Duration::ZERO,
                    made_reds: 0,
                    cands: vec![],
                    reds: vec![],
                    red_index: HashMap::new(),
                    order_jobs,
                    yields: HashMap::new(),
                    prev_yields: HashMap::new(),
                };
                run.level.jobs = run.level_jobs(run.level.len, false);
                // The order applies once a solution exists (a 60-move one keeps every
                // candidate).
                run.best = Some(Solution {
                    moves: vec![0; 60],
                    phases: [0; 3],
                    elapsed: Duration::ZERO,
                });
                let mut want = 64;
                while run.level.len <= last {
                    run.phase1(want);
                    want *= 2;
                }
                let mut c: Vec<(Vec<u8>, usize)> = run
                    .cands
                    .iter()
                    .filter(|c| c.moves1.len() <= last as usize)
                    .map(|c| (c.moves1.clone(), c.frame))
                    .collect();
                c.sort();
                found.push(c);
                // The yields of length `last` (4-move jobs when ordered, else 3-move ones
                // counting their 4-move subtrees) are the same.
                ys.push(run.prev_yields.clone());
                if order_jobs {
                    let plain = run.level_jobs(last + 1, true);
                    assert!(plain.iter().all(|j| j.1.len() == 4));
                    let mut o = plain.clone();
                    order_by_yield(&mut o, &run.prev_yields);
                    assert_ne!(o, plain);
                    o.sort();
                    let mut p = plain;
                    p.sort();
                    assert_eq!(o, p);
                }
            }
            assert!(
                found[0].len() > 100,
                "too few candidates ({})",
                found[0].len()
            );
            assert_eq!(found[0], found[1]);
            assert!(ys[0].len() > 1000, "too few prefixes ({})", ys[0].len());
            assert_eq!(ys[0], ys[1]);
        }
    }

    /// A phase-2 level with every search split gives exactly the reductions of the unsplit
    /// searches, in the same order.
    #[test]
    fn split_phase2_gives_the_same_reductions() {
        let dir = std::path::Path::new("tables");
        let t = load_tables_with(dir, 3, true, false, (None, None), 8, false);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(13);
        let mut total = 0;
        for _ in 0..3 {
            let cube = Cube::random(&mut rng);
            let (stop, have) = (AtomicBool::new(false), AtomicBool::new(false));
            let mut report = |_: &Solution| {};
            let root = P1State::from_cube(&cube);
            let pairs = vec![crate::phase1::ALL_PAIRS];
            let mut d0 = [0u8; 4];
            d0[pairs[0]] = t.p1.distance_pair(&root, pairs[0]);
            let start = Instant::now();
            let mut run = Run {
                t: &t,
                cube,
                threads: 4,
                stop: &stop,
                have: &have,
                start,
                limit: start + Duration::from_secs(3600),
                p1_cut: None,
                p2_cut: None,
                split2: u8::MAX,
                best: None,
                report: &mut report,
                target: None,
                stats: Stats::default(),
                root,
                d0,
                pairs,
                level: Level1 {
                    len: d0[crate::phase1::ALL_PAIRS],
                    jobs: vec![],
                    next: 0,
                },
                made_cands: 0,
                cand_time: Duration::ZERO,
                made_reds: 0,
                cands: vec![],
                reds: vec![],
                red_index: HashMap::new(),
                order_jobs: true,
                yields: HashMap::new(),
                prev_yields: HashMap::new(),
            };
            run.level.jobs = run.level_jobs(run.level.len, false);
            run.phase1(1000);
            // As if a 60-move solution existed: phase 2 keeps every solution of a search.
            run.best = Some(Solution {
                moves: vec![0; 60],
                phases: [0; 3],
                elapsed: Duration::ZERO,
            });
            let lo = run
                .cands
                .iter()
                .map(|c| c.moves1.len() + c.h2 as usize)
                .min()
                .unwrap();
            let mut found = 0;
            for t12 in lo..lo + 8 {
                if found > 2000 {
                    break;
                }
                let done: Vec<u8> = run.cands.iter().map(|c| c.done2).collect();
                let key = |r: &Reduction| (r.frame, r.moves12.clone(), r.c3, r.h3);
                run.split2 = u8::MAX;
                let whole: Vec<_> = run.phase2_level(t12, usize::MAX).iter().map(key).collect();
                let done_whole: Vec<u8> = run.cands.iter().map(|c| c.done2).collect();
                for (c, &d) in run.cands.iter_mut().zip(&done) {
                    c.done2 = d;
                }
                run.split2 = 0;
                let split: Vec<_> = run.phase2_level(t12, usize::MAX).iter().map(key).collect();
                assert_eq!(split, whole, "level {t12}");
                let done_split: Vec<u8> = run.cands.iter().map(|c| c.done2).collect();
                assert_eq!(done_split, done_whole);
                found += whole.len();
            }
            total += found;
        }
        assert!(total > 3000, "too few reductions to test ({total})");
    }

    /// Jobs cut short at random points, with their handed-back subtrees searched later, find
    /// exactly the endpoints of one uninterrupted search.
    #[test]
    fn split_jobs_find_the_same_endpoints() {
        let t = load_p1(std::path::Path::new("tables"), false, 8, false);
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
                    let close = Close::new(None);
                    let (st, d) = root.after(&prefix, d0, &t);
                    let mut spill = vec![];
                    let mut srch = Search1::new(&t, |p: &[u8], f| {
                        got.push((p.to_vec(), f));
                        count += 1;
                        if count % 3 == scramble % 3 {
                            close.raise();
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
                        &mut vec![],
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
