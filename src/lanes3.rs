//! Phase 3 with lanes: each thread interleaves several independent searches (phase-3 tasks),
//! node by node. A lane expands a node (children generated, table entries prefetched) and
//! yields; by the time it resumes and reads those entries, the other lanes have run, so the
//! memory latency of the table lookups is hidden (egg-solver's micro-threading).
//!
//! The search itself is exactly that of [`crate::phase3::Search3`]: the same children, bounds,
//! side choice and move order.

use crate::cube3::N_MOVES3;
use crate::moves::ALLOWED_AFTER;
use crate::phase3::{ALLOWED_BEFORE, KNOWN, Node, P3Tables, QUARTER, Task};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
struct Child {
    node: Node,
    m: u8,
    iv: usize,
    iy: usize,
    hv: u8,
    hy: u8,
}

struct Frame {
    node: Node,
    left: u8,
    last_f: usize,
    last_b: usize,
    hv: u8,
    hy: u8,
    /// The move that led here went to the back of the solution (else the front; none at the
    /// root of a task).
    via_back: bool,
    expanded: bool,
    back: bool,
    kids: [MaybeUninit<Child>; 18],
    nk: u8,
    next: u8,
}

struct Lane {
    job: usize,
    /// Preallocated frames; `frames[..top]` is the stack.
    frames: Box<[Frame]>,
    top: usize,
    front: Vec<u8>,
    back: Vec<u8>,
}

fn empty_frame() -> Frame {
    Frame {
        node: Node {
            v: crate::simd::V32::ZERO,
            y: crate::simd::V32::ZERO,
        },
        left: 0,
        last_f: 0,
        last_b: 0,
        hv: 0,
        hy: 0,
        via_back: false,
        expanded: false,
        back: false,
        kids: [const { MaybeUninit::uninit() }; 18],
        nk: 0,
        next: 0,
    }
}

impl Lane {
    fn new() -> Lane {
        Lane {
            job: 0,
            frames: (0..24).map(|_| empty_frame()).collect(),
            top: 0,
            front: vec![],
            back: vec![],
        }
    }

    /// Push a frame (not yet expanded) for a node.
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        node: Node,
        left: u8,
        last_f: usize,
        last_b: usize,
        hv: u8,
        hy: u8,
        via_back: bool,
    ) {
        let f = &mut self.frames[self.top];
        f.node = node;
        f.left = left;
        f.last_f = last_f;
        f.last_b = last_b;
        f.hv = hv;
        f.hy = hy;
        f.via_back = via_back;
        f.expanded = false;
        self.top += 1;
    }

    fn reset(&mut self, task: &Task, job: usize) {
        self.job = job;
        self.top = 0;
        self.front.clone_from(&task.front);
        self.back.clone_from(&task.back);
        self.push(
            task.node,
            task.left,
            task.last_f,
            task.last_b,
            task.hv,
            task.hy,
            false,
        );
    }
}

/// Outcome of one job.
pub enum Outcome {
    /// Searched completely, no solution.
    Done(usize),
    /// A solution (full move sequence).
    Found(usize, Vec<u8>),
}

pub struct Lanes<'a> {
    t: &'a P3Tables,
    lanes: Vec<Lane>,
    active: Vec<bool>,
    pub nodes: u64,
}

impl<'a> Lanes<'a> {
    pub fn new(t: &'a P3Tables, n: usize) -> Lanes<'a> {
        Lanes {
            t,
            lanes: (0..n).map(|_| Lane::new()).collect(),
            active: vec![false; n],
            nodes: 0,
        }
    }

    /// Run jobs from `next_job` (returning `None` when there are no more) through the lanes,
    /// reporting each outcome; stops early when `stop` is raised.
    pub fn run(
        &mut self,
        mut next_job: impl FnMut() -> Option<(usize, &'a Task)>,
        stop: &AtomicBool,
        mut report: impl FnMut(Outcome),
    ) {
        let n = self.lanes.len();
        let mut live = 0;
        for i in 0..n {
            self.active[i] = false;
            if let Some((j, task)) = next_job() {
                self.lanes[i].reset(task, j);
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
                if let Some(out) = step(self.t, &mut self.lanes[i], &mut self.nodes) {
                    report(out);
                    match next_job() {
                        Some((j, task)) => self.lanes[i].reset(task, j),
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

/// Advance a lane until it has issued the prefetches of one expansion, or finished its job.
fn step(t: &P3Tables, lane: &mut Lane, nodes: &mut u64) -> Option<Outcome> {
    let outer = (1u32 << N_MOVES3) - 1;
    loop {
        if lane.top == 0 {
            return Some(Outcome::Done(lane.job));
        }
        let top = lane.top - 1;
        let f = &mut lane.frames[top];
        if !f.expanded {
            let left = f.left;
            let tight = f.hv == left || f.hy == left;
            if left == 1 {
                let mut allowed = ALLOWED_AFTER[f.last_f]
                    & ALLOWED_BEFORE[f.last_b]
                    & outer
                    & if tight { QUARTER } else { u32::MAX };
                while allowed != 0 {
                    let m = allowed.trailing_zeros() as usize;
                    allowed &= allowed - 1;
                    *nodes += 1;
                    if f.node.v.apply(m).is_solved() {
                        let mut p = lane.front.clone();
                        p.push(m as u8);
                        p.extend(lane.back.iter().rev());
                        return Some(Outcome::Found(lane.job, p));
                    }
                }
                pop(lane);
                continue;
            }
            // Expansion (stage 1 of Search3::dfs): children, corner bounds, prefetches.
            let back = f.hv == left && f.hy != left;
            f.back = back;
            f.expanded = true;
            let n = f.node;
            let (hv, hy) = (f.hv, f.hy);
            let mut nk = 0usize;
            let mut allowed = if back {
                ALLOWED_BEFORE[f.last_b]
            } else {
                ALLOWED_AFTER[f.last_f]
            } & outer
                & if tight { QUARTER } else { u32::MAX };
            while allowed != 0 {
                let face = allowed.trailing_zeros() / 3;
                let fm = allowed >> (face * 3) & 7;
                allowed &= !(7 << (face * 3));
                let mut twin: Option<(u8, usize)> = None;
                for pw in 0..3 {
                    if fm >> pw & 1 == 0 {
                        continue;
                    }
                    let m = (face * 3 + pw) as usize;
                    *nodes += 1;
                    let (node, own, other) = if back {
                        let b = crate::moves::inverse_move(m);
                        let node = Node {
                            v: n.v.apply_inv(b),
                            y: n.y.apply(b),
                        };
                        (node, node.y, node.v)
                    } else {
                        let node = Node {
                            v: n.v.apply(m),
                            y: n.y.apply_inv(m),
                        };
                        (node, node.v, node.y)
                    };
                    let (oc, oi) = t.probe(&own);
                    if oc >= left {
                        continue;
                    }
                    let (xc, xi, xh) = if pw == 1 {
                        (0, KNOWN, if back { hv } else { hy })
                    } else {
                        match twin {
                            Some((c, i)) => (c, i, 0),
                            None => {
                                let (c, i) = t.probe(&other);
                                twin = Some((c, i));
                                (c, i, 0)
                            }
                        }
                    };
                    if xc >= left || xh >= left {
                        continue;
                    }
                    t.h48.prefetch(oi);
                    if xi != KNOWN {
                        t.h48.prefetch(xi);
                    }
                    let (iv, iy, hv0, hy0) = if back {
                        (xi, oi, xc.max(xh), oc)
                    } else {
                        (oi, xi, oc, xc.max(xh))
                    };
                    f.kids[nk].write(Child {
                        node,
                        m: m as u8,
                        iv,
                        iy,
                        hv: hv0,
                        hy: hy0,
                    });
                    nk += 1;
                }
            }
            f.nk = nk as u8;
            f.next = 0;
            return None;
        }
        // Expanded: the next child that survives its bounds becomes the new top.
        let r = f.left - 1;
        let mut pushed = None;
        while f.next < f.nk {
            let k = unsafe { f.kids[f.next as usize].assume_init_ref() };
            f.next += 1;
            let cv = if k.iv == KNOWN {
                k.hv
            } else {
                t.h48.value(k.iv).max(k.hv)
            };
            if cv > r {
                continue;
            }
            let cy = if k.iy == KNOWN {
                k.hy
            } else {
                t.h48.value(k.iy).max(k.hy)
            };
            if cy > r {
                continue;
            }
            let (lf, lb) = if f.back {
                (f.last_f, k.m as usize)
            } else {
                (k.m as usize, f.last_b)
            };
            pushed = Some((k.node, lf, lb, cv, cy, f.back, k.m));
            break;
        }
        match pushed {
            Some((node, lf, lb, cv, cy, via_back, m)) => {
                if via_back {
                    lane.back.push(m);
                } else {
                    lane.front.push(m);
                }
                lane.push(node, r, lf, lb, cv, cy, via_back);
            }
            None => pop(lane),
        }
    }
}

fn pop(lane: &mut Lane) {
    lane.top -= 1;
    if lane.top > 0 {
        if lane.frames[lane.top].via_back {
            lane.back.pop();
        } else {
            lane.front.pop();
        }
    }
}
