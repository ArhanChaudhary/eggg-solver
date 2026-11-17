//! Phase 1 with lanes: each thread interleaves several phase-1 jobs (prefix subtrees) node by
//! node. A lane expands a node (children generated, staging-table entries prefetched) and
//! yields; when it resumes, the other lanes have run and the entries are in cache.
//!
//! The search is exactly that of [`crate::phase1::Search1`]: same children, pruning and
//! endpoint tests, so the same endpoints are found.

use crate::moves::{ALLOWED_AFTER, N_MOVES, WIDE_QUARTER};
use crate::phase1::{P1State, P1Tables};
use crate::table::Mod3Table;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};

struct Frame {
    s: P1State,
    d: u8,
    left: u8,
    last: usize,
    expanded: bool,
    kids: [MaybeUninit<(P1State, u64, u8)>; N_MOVES],
    nk: u8,
    next: u8,
}

fn empty_frame(s: P1State) -> Frame {
    Frame {
        s,
        d: 0,
        left: 0,
        last: 0,
        expanded: false,
        kids: [const { MaybeUninit::uninit() }; N_MOVES],
        nk: 0,
        next: 0,
    }
}

struct Lane {
    job: usize,
    frames: Box<[Frame]>,
    top: usize,
    path: Vec<u8>,
    base: usize,
}

/// A phase-1 job: the subtree below a prefix.
pub struct Job<'j> {
    pub state: P1State,
    pub d: u8,
    pub left: u8,
    pub prefix: &'j [u8],
}

impl Lane {
    fn new(s: P1State) -> Lane {
        Lane {
            job: 0,
            frames: (0..32).map(|_| empty_frame(s)).collect(),
            top: 0,
            path: Vec::with_capacity(40),
            base: 0,
        }
    }

    fn push(&mut self, s: P1State, d: u8, left: u8, last: usize) {
        let f = &mut self.frames[self.top];
        f.s = s;
        f.d = d;
        f.left = left;
        f.last = last;
        f.expanded = false;
        self.top += 1;
    }

    fn reset(&mut self, job: &Job, id: usize) {
        self.job = id;
        self.top = 0;
        self.path.clear();
        self.path.extend_from_slice(job.prefix);
        self.base = job.prefix.len();
        let last = job.prefix.last().map(|&m| m as usize).unwrap_or(N_MOVES);
        self.push(job.state, job.d, job.left, last);
    }

    fn pop(&mut self) {
        self.top -= 1;
        if self.path.len() > self.base {
            self.path.pop();
        }
    }
}

pub struct Lanes1<'a> {
    t: &'a P1Tables,
    lanes: Vec<Lane>,
    active: Vec<bool>,
    pub nodes: u64,
}

impl<'a> Lanes1<'a> {
    pub fn new(t: &'a P1Tables, n: usize, any: P1State) -> Lanes1<'a> {
        Lanes1 {
            t,
            lanes: (0..n).map(|_| Lane::new(any)).collect(),
            active: vec![false; n],
            nodes: 0,
        }
    }

    /// Run jobs from `next_job` through the lanes; `endpoint(job id, path, frames)` for each
    /// endpoint, `done(job id)` when a job has been searched completely.
    pub fn run<'j>(
        &mut self,
        mut next_job: impl FnMut() -> Option<(usize, Job<'j>)>,
        stop: &AtomicBool,
        mut endpoint: impl FnMut(usize, &[u8], u8),
        mut done: impl FnMut(usize),
    ) {
        let n = self.lanes.len();
        let mut live = 0;
        for i in 0..n {
            self.active[i] = false;
            if let Some((id, job)) = next_job() {
                self.lanes[i].reset(&job, id);
                self.active[i] = true;
                live += 1;
            }
        }
        let mut steps = 0u64;
        while live > 0 {
            for i in 0..n {
                if !self.active[i] {
                    continue;
                }
                let lane = &mut self.lanes[i];
                if step(self.t, lane, &mut self.nodes, &mut endpoint) {
                    done(lane.job);
                    match next_job() {
                        Some((id, job)) => self.lanes[i].reset(&job, id),
                        None => {
                            self.active[i] = false;
                            live -= 1;
                        }
                    }
                }
            }
            steps += 1;
            if steps & 255 == 0 && stop.load(Ordering::Relaxed) {
                return;
            }
        }
    }
}

/// Advance a lane until it has issued the prefetches of one expansion (false), or finished its
/// job (true).
fn step(
    t: &P1Tables,
    lane: &mut Lane,
    nodes: &mut u64,
    endpoint: &mut impl FnMut(usize, &[u8], u8),
) -> bool {
    loop {
        if lane.top == 0 {
            return true;
        }
        let top = lane.top - 1;
        let f = &mut lane.frames[top];
        if !f.expanded {
            let (s, d, left, last) = (f.s, f.d, f.left, f.last);
            if left == 0 {
                if d == 0 {
                    let frames = t.endpoint_frames(&s, last);
                    if frames != 0 {
                        endpoint(lane.job, &lane.path, frames);
                    }
                }
                lane.pop();
                continue;
            }
            if d == 0 && s.parity && left <= 2 {
                lane.pop();
                continue;
            }
            let mut allowed = ALLOWED_AFTER[last];
            if left == 1 {
                // Last move: parity and wings first, then the centres (as in Search1).
                allowed &= if s.parity {
                    WIDE_QUARTER
                } else {
                    !WIDE_QUARTER
                };
                while allowed != 0 {
                    let m = allowed.trailing_zeros() as usize;
                    allowed &= allowed - 1;
                    *nodes += 1;
                    let pos = t.wing_dst[m].gather(&s.pos);
                    let mut frames = 0u8;
                    for (fi, fr) in t.frames.iter().enumerate() {
                        if fr.p2_mask >> m & 1 == 0 && fr.wings_ok(&pos) {
                            frames |= 1 << fi;
                        }
                    }
                    if frames == 0 {
                        continue;
                    }
                    let c = s.apply(m, t);
                    if d != 0 && Mod3Table::decode(d, t.value(&c)) != 0 {
                        continue;
                    }
                    for (fi, fr) in t.frames.iter().enumerate() {
                        if frames >> fi & 1 == 1 && !fr.centres_ok(&c.centres) {
                            frames &= !(1 << fi);
                        }
                    }
                    if frames != 0 {
                        lane.path.push(m as u8);
                        endpoint(lane.job, &lane.path, frames);
                        lane.path.pop();
                    }
                }
                lane.pop();
                continue;
            }
            // Expansion: children, table indices, prefetches.
            let f = &mut lane.frames[top];
            let mut nk = 0;
            while allowed != 0 {
                let m = allowed.trailing_zeros() as usize;
                allowed &= allowed - 1;
                let c = s.apply(m, t);
                let (ud, rl) = c.masks();
                let i = if t.big {
                    t.staging.big_index(ud, rl)
                } else {
                    t.staging.fast_index(unsafe { *t.staging.info(ud) }, rl)
                };
                t.table.prefetch(i);
                f.kids[nk].write((c, i, m as u8));
                nk += 1;
            }
            *nodes += nk as u64;
            f.nk = nk as u8;
            f.next = 0;
            f.expanded = true;
            return false;
        }
        // Expanded: descend into the next child that can still reach the goal.
        let mut pushed = None;
        while f.next < f.nk {
            let (c, i, m) = unsafe { *f.kids[f.next as usize].assume_init_ref() };
            f.next += 1;
            let dc = Mod3Table::decode(f.d, t.table.get(i));
            if dc < f.left {
                pushed = Some((c, dc, f.left - 1, m));
                break;
            }
        }
        match pushed {
            Some((c, dc, l, m)) => {
                lane.path.push(m);
                lane.push(c, dc, l, m as usize);
            }
            None => lane.pop(),
        }
    }
}
