//! Phase 3: solve the reduced 3x3x3 with the 18 outer face turns, in one IDA* search.
//!
//! Heuristic: the H48 table (with its corner-class bound) of the cube and of its inverse, in a
//! bidirectional search that extends the side whose next move is restricted.

use crate::c3simd::{C3, H48Fast};
use crate::cube3::{Cube3, N_MOVES3};
use crate::h48::{Coords, DEFAULT_BASE, H48};
use crate::moves::ALLOWED_AFTER;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct P3Tables {
    pub coords: Coords,
    pub fast: H48Fast,
    pub h48: H48,
}

impl P3Tables {
    pub fn load_or_build(dir: &std::path::Path, k: u32, threads: usize, verbose: bool) -> P3Tables {
        let coords = Coords::new();
        let base = DEFAULT_BASE[k as usize];
        let name = format!("p3-h48-h{k}b{base}");
        let path = dir.join(format!("{name}.tab"));
        let mut h48 = H48 {
            k,
            base,
            data: crate::table::HugeBuf::zeroed(H48::bytes(k) / 8),
        };
        let bytes = h48.data.as_bytes_mut();
        if crate::table::load(&path, &name, bytes).is_err() {
            if verbose {
                eprintln!(
                    "generating phase-3 h48 table h{k} (base {base}, {:.1} MB)...",
                    H48::bytes(k) as f64 / 1e6
                );
            }
            h48 = H48::generate(&coords, k, base, threads, verbose);
            let bytes = h48.data.as_bytes();
            crate::table::save(&path, &name, bytes).expect("saving table");
        }
        let fast = H48Fast::new(&coords);
        P3Tables { coords, fast, h48 }
    }
}

/// Quarter turns among the 18 outer moves.
const QUARTER: u32 = {
    let mut m = 0u32;
    let mut i = 0;
    while i < 18 {
        if i % 3 != 1 {
            m |= 1 << i;
        }
        i += 1;
    }
    m
};

/// Moves that may precede move `b` in a canonical sequence (`N_MOVES` = none follows).
const ALLOWED_BEFORE: [u32; crate::moves::N_MOVES + 1] = {
    let mut t = [0u32; crate::moves::N_MOVES + 1];
    let mut b = 0;
    while b <= crate::moves::N_MOVES {
        let mut a = 0;
        while a < crate::moves::N_MOVES {
            if b == crate::moves::N_MOVES || ALLOWED_AFTER[a] >> b & 1 == 1 {
                t[b] |= 1 << a;
            }
            a += 1;
        }
        b += 1;
    }
    t
};

/// A phase-3 node: the cube and its inverse.
#[derive(Clone, Copy)]
struct Node {
    v: C3,
    y: C3,
}

const KNOWN: usize = usize::MAX;

struct Kid {
    node: Node,
    m: u8,
    /// Table indices still to read (KNOWN: the value is already in `hv` / `hy`).
    iv: usize,
    iy: usize,
    hv: u8,
    hy: u8,
}

impl P3Tables {
    /// (corner-class distance, h48 index) of a cube.
    #[inline(always)]
    fn probe(&self, v: &C3) -> (u8, usize) {
        let i = self.fast.cocsep(v);
        debug_assert!(i < self.coords.cocsep.len());
        // SAFETY: a cube's cocsep index is below N_COCSEP (the table's length).
        let cs = unsafe { *self.coords.cocsep.get_unchecked(i) };
        let (eslot, msplit, eo) = self.fast.edges(v, (cs >> 8 & 255) as usize);
        (
            (cs & 255) as u8,
            self.h48.index(cs >> 16, eslot, msplit, eo),
        )
    }

    /// Bound of a cube (debugging aid).
    pub fn bound_pub(&self, x: &Cube3) -> u8 {
        self.bound(&C3::from_cube(x))
    }

    fn bound(&self, v: &C3) -> u8 {
        let (a, i) = self.probe(v);
        a.max(self.h48.value(i))
    }
}

pub struct Search3<'a> {
    pub t: &'a P3Tables,
    front: Vec<u8>,
    back: Vec<u8>,
    pub nodes: u64,
    pub abort: Option<&'a AtomicBool>,
    /// A second abort flag (another search of the same level found a solution).
    pub abort_level: Option<&'a AtomicBool>,
    pub aborted: bool,
    /// When set, the root's surviving children are collected as tasks instead of searched.
    collect: Option<(u8, Vec<Task>)>,
}

/// A subtree of a phase-3 search (one child of the root), to be searched separately.
#[derive(Clone)]
pub struct Task {
    node: Node,
    left: u8,
    last_f: usize,
    last_b: usize,
    hv: u8,
    hy: u8,
    front: Vec<u8>,
    back: Vec<u8>,
}

impl<'a> Search3<'a> {
    pub fn new(t: &'a P3Tables) -> Self {
        Search3 {
            t,
            front: Vec::with_capacity(24),
            back: Vec::with_capacity(24),
            nodes: 0,
            abort: None,
            abort_level: None,
            aborted: false,
            collect: None,
        }
    }

    pub fn lower_bound(&self, c: &Cube3) -> u8 {
        let v = C3::from_cube(c);
        self.t.bound(&v).max(self.t.bound(&v.inverse()))
    }

    /// Look for a solution of exactly `depth` moves (callers search depths in increasing
    /// order).
    pub fn search_depth(&mut self, c: &Cube3, depth: u8) -> Option<Vec<u8>> {
        let v = C3::from_cube(c);
        let n = Node { v, y: v.inverse() };
        let (hv, hy) = (self.t.bound(&n.v), self.t.bound(&n.y));
        if hv.max(hy) > depth {
            return None;
        }
        if depth == 0 {
            return v.is_solved().then(Vec::new);
        }
        self.front.clear();
        self.back.clear();
        if self.dfs(
            &n,
            depth,
            crate::moves::N_MOVES,
            crate::moves::N_MOVES,
            hv,
            hy,
        ) {
            let mut p = self.front.clone();
            p.extend(self.back.iter().rev());
            Some(p)
        } else {
            None
        }
    }

    /// The subtrees of the search of exactly `depth` moves (one per surviving child of the
    /// root), or `None` if the root itself is pruned. Searching every task is equivalent to
    /// [`Search3::search_depth`] (with no restriction on the first move).
    pub fn tasks(&mut self, c: &Cube3, depth: u8) -> Option<Vec<Task>> {
        let v = C3::from_cube(c);
        let n = Node { v, y: v.inverse() };
        let (hv, hy) = (self.t.bound(&n.v), self.t.bound(&n.y));
        if hv.max(hy) > depth || depth < 2 {
            return None;
        }
        self.front.clear();
        self.back.clear();
        self.collect = Some((depth, vec![]));
        self.dfs(
            &n,
            depth,
            crate::moves::N_MOVES,
            crate::moves::N_MOVES,
            hv,
            hy,
        );
        self.collect.take().map(|c| c.1)
    }

    /// Search one task; the full solution if found.
    pub fn run_task(&mut self, task: &Task) -> Option<Vec<u8>> {
        self.front.clone_from(&task.front);
        self.back.clone_from(&task.back);
        if self.dfs(
            &task.node,
            task.left,
            task.last_f,
            task.last_b,
            task.hv,
            task.hy,
        ) {
            let mut p = self.front.clone();
            p.extend(self.back.iter().rev());
            Some(p)
        } else {
            None
        }
    }

    /// Shortest solution of at most `max_len` moves.
    pub fn solve(&mut self, c: &Cube3, max_len: u8) -> Option<Vec<u8>> {
        let h = self.lower_bound(c);
        (h..=max_len).find_map(|d| self.search_depth(c, d))
    }

    /// The solution is front moves, then `left` unknown moves, then the back moves reversed;
    /// `n.v` is the cube with the front moves applied and the back moves undone, so the
    /// unknown middle solves it. `hv`/`hy` are lower bounds of the middle from both sides.
    ///
    /// A tight bound on the cube means the middle's last move is a quarter turn (the target
    /// group contains the half turns), a tight bound on the inverse that its first move is: so
    /// the search extends the side whose next move is restricted.
    fn dfs(&mut self, n: &Node, left: u8, last_f: usize, last_b: usize, hv: u8, hy: u8) -> bool {
        if self.aborted {
            return false;
        }
        if self.nodes & 4095 == 0
            && (self.abort.is_some_and(|a| a.load(Ordering::Relaxed))
                || self.abort_level.is_some_and(|a| a.load(Ordering::Relaxed)))
        {
            self.aborted = true;
            return false;
        }
        let t = self.t;
        let outer = (1u32 << N_MOVES3) - 1;
        let tight = hv == left || hy == left;
        let back = hv == left && hy != left;
        if left == 1 {
            let mut allowed = ALLOWED_AFTER[last_f]
                & ALLOWED_BEFORE[last_b]
                & outer
                & if tight { QUARTER } else { u32::MAX };
            while allowed != 0 {
                let m = allowed.trailing_zeros() as usize;
                allowed &= allowed - 1;
                self.nodes += 1;
                if n.v.apply_c(&t.fast.moves[m]).is_solved() {
                    self.front.push(m as u8);
                    return true;
                }
            }
            return false;
        }
        let mut kids: [std::mem::MaybeUninit<Kid>; 18] =
            [const { std::mem::MaybeUninit::uninit() }; 18];
        let mut nk = 0;
        // Stage 1: children, corner-class bounds, prefetch of the table entries.
        let mut allowed = if back {
            ALLOWED_BEFORE[last_b]
        } else {
            ALLOWED_AFTER[last_f]
        } & outer
            & if tight { QUARTER } else { u32::MAX };
        while allowed != 0 {
            let face = allowed.trailing_zeros() / 3;
            let fm = allowed >> (face * 3) & 7;
            allowed &= !(7 << (face * 3));
            // The side not moved by a turn of this face gets the same coset for X and X'
            // (X2 is in the target group), and its parent's for X2.
            let mut twin: Option<(u8, usize)> = None;
            for pw in 0..3 {
                if fm >> pw & 1 == 0 {
                    continue;
                }
                let m = (face * 3 + pw) as usize;
                self.nodes += 1;
                let (node, own, other) = if back {
                    let b = &t.fast.moves[crate::moves::inverse_move(m)];
                    let node = Node {
                        v: n.v.apply_inv_c(b),
                        y: n.y.apply_c(b),
                    };
                    (node, node.y, node.v)
                } else {
                    let c = &t.fast.moves[m];
                    let node = Node {
                        v: n.v.apply_c(c),
                        y: n.y.apply_inv_c(c),
                    };
                    (node, node.v, node.y)
                };
                // `own`: the side that moved on the right (its coset changes); `other` moved
                // on the left.
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
                kids[nk].write(Kid {
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
        // Stage 2: read the bounds and descend.
        let r = left - 1;
        let back_side = back;
        for k in kids.iter().take(nk) {
            let k = unsafe { k.assume_init_ref() };
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
            if let Some((root, tasks)) = &mut self.collect
                && *root == left
            {
                let (mut front, mut back) = (self.front.clone(), self.back.clone());
                let (lf, lb) = if back_side {
                    (last_f, k.m as usize)
                } else {
                    (k.m as usize, last_b)
                };
                if back_side {
                    back.push(k.m);
                } else {
                    front.push(k.m);
                }
                tasks.push(Task {
                    node: k.node,
                    left: r,
                    last_f: lf,
                    last_b: lb,
                    hv: cv,
                    hy: cy,
                    front,
                    back,
                });
                continue;
            }
            let found = if back {
                self.back.push(k.m);
                self.dfs(&k.node, r, last_f, k.m as usize, cv, cy)
            } else {
                self.front.push(k.m);
                self.dfs(&k.node, r, k.m as usize, last_b, cv, cy)
            };
            if found {
                return true;
            }
            if back {
                self.back.pop();
            } else {
                self.front.pop();
            }
        }
        false
    }
}
