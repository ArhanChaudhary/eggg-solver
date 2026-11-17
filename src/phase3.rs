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
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                h48.data.as_mut_slice().as_mut_ptr() as *mut u8,
                H48::bytes(k),
            )
        };
        if crate::table::load(&path, &name, bytes).is_err() {
            if verbose {
                eprintln!(
                    "generating phase-3 h48 table h{k} (base {base}, {:.1} MB)...",
                    H48::bytes(k) as f64 / 1e6
                );
            }
            h48 = H48::generate(&coords, k, base, threads, verbose);
            let bytes = unsafe {
                std::slice::from_raw_parts(h48.data.as_slice().as_ptr() as *const u8, H48::bytes(k))
            };
            crate::table::save(&path, &name, bytes).expect("saving table");
        }
        let fast = H48Fast::new(&coords);
        P3Tables { coords, fast, h48 }
    }

    /// H48 lower bound of a cube (with the corner-class bound).
    #[inline]
    pub fn h48_bound(&self, x: &Cube3) -> u8 {
        let co = &self.coords;
        let cs = co.cocsep[co.cocsep_of(x)];
        let class = cs >> 16;
        let tt = (cs >> 8 & 255) as usize;
        let cd = (cs & 255) as u8;
        let (eslot, msplit, eo) = co.edge_part_conj(x, tt);
        let v = self.h48.value(self.h48.index(class, eslot, msplit, eo));
        v.max(cd)
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
        let cs = self.coords.cocsep[self.fast.cocsep(v)];
        (cs as u8, self.probe_edges(v, cs))
    }

    /// The cocsep entry (class << 16 | symmetry << 8 | corner distance) of a cube.
    #[inline(always)]
    fn corners(&self, v: &C3) -> u32 {
        self.coords.cocsep[self.fast.cocsep(v)]
    }

    /// The h48 index of a cube, given its cocsep entry.
    #[inline(always)]
    fn probe_edges(&self, v: &C3, cs: u32) -> usize {
        let (eslot, msplit, eo) = self.fast.edges(v, (cs >> 8 & 255) as usize);
        self.h48.index(cs >> 16, eslot, msplit, eo)
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
    pub node_limit: u64,
    pub abort: Option<&'a AtomicBool>,
    /// A second abort flag (another search of the same level found a solution).
    pub abort_level: Option<&'a AtomicBool>,
    pub aborted: bool,
}

impl<'a> Search3<'a> {
    pub fn new(t: &'a P3Tables) -> Self {
        Search3 {
            t,
            front: Vec::with_capacity(24),
            back: Vec::with_capacity(24),
            nodes: 0,
            node_limit: u64::MAX,
            abort: None,
            abort_level: None,
            aborted: false,
        }
    }

    pub fn lower_bound(&self, c: &Cube3) -> u8 {
        let v = C3::from_cube(c);
        self.t.bound(&v).max(self.t.bound(&v.inverse()))
    }

    /// Look for a solution of exactly `depth` moves whose first move may follow `first_after`
    /// (callers search depths in increasing order).
    pub fn search_depth(&mut self, c: &Cube3, depth: u8, first_after: usize) -> Option<Vec<u8>> {
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
            first_after,
            crate::moves::N_MOVES,
            hv,
            hy,
            hy == depth,
            hv == depth,
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
        (h..=max_len).find_map(|d| self.search_depth(c, d, crate::moves::N_MOVES))
    }

    /// The solution is front moves, then `left` unknown moves, then the back moves reversed;
    /// `n.v` is the cube with the front moves applied and the back moves undone, so the
    /// unknown middle solves it. `hv`/`hy` are lower bounds of the middle from both sides.
    ///
    /// A tight bound on the cube means the middle's last move is a quarter turn (the target
    /// group contains the half turns), a tight bound on the inverse that its first move is.
    /// These facts hold until that end of the middle is extended (`fq`: the next front move
    /// must be a quarter turn, `bq`: the next back move), and the search extends whichever
    /// side has fewer moves to try.
    #[allow(clippy::too_many_arguments)]
    fn dfs(
        &mut self,
        n: &Node,
        left: u8,
        last_f: usize,
        last_b: usize,
        hv: u8,
        hy: u8,
        fq: bool,
        bq: bool,
    ) -> bool {
        if self.aborted {
            return false;
        }
        if self.nodes > self.node_limit
            || (self.nodes & 4095 == 0
                && (self.abort.is_some_and(|a| a.load(Ordering::Relaxed))
                    || self.abort_level.is_some_and(|a| a.load(Ordering::Relaxed))))
        {
            self.aborted = true;
            return false;
        }
        let t = self.t;
        let outer = (1u32 << N_MOVES3) - 1;
        let fmask = ALLOWED_AFTER[last_f] & outer & if fq { QUARTER } else { u32::MAX };
        let bmask = ALLOWED_BEFORE[last_b] & outer & if bq { QUARTER } else { u32::MAX };
        if left == 1 {
            let mut allowed = fmask & bmask;
            while allowed != 0 {
                let m = allowed.trailing_zeros() as usize;
                allowed &= allowed - 1;
                self.nodes += 1;
                if n.v.apply(m).is_solved() {
                    self.front.push(m as u8);
                    return true;
                }
            }
            return false;
        }
        let back = bmask.count_ones() < fmask.count_ones();
        let r = left - 1;
        // The table cannot prune with more than base + 3 moves left.
        let read = r <= t.h48.base + 3;
        let mut kids: [std::mem::MaybeUninit<Kid>; 18] =
            [const { std::mem::MaybeUninit::uninit() }; 18];
        let mut nk = 0;
        // Stage 1: children, corner-class bounds, prefetch of the table entries.
        let mut allowed = if back { bmask } else { fmask };
        while allowed != 0 {
            let face = allowed.trailing_zeros() / 3;
            let fm = allowed >> (face * 3) & 7;
            allowed &= !(7 << (face * 3));
            // The side not moved on the right gets the same coset for X and X' (X2 is in the
            // target group), and its parent's for X2.
            let mut twin: Option<(u8, usize)> = None;
            for pw in 0..3 {
                if fm >> pw & 1 == 0 {
                    continue;
                }
                let m = (face * 3 + pw) as usize;
                self.nodes += 1;
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
                let ocs = t.corners(&own);
                let oc = ocs as u8;
                if oc > r {
                    continue;
                }
                let (xc, xi, xh) = if pw == 1 {
                    (0, KNOWN, if back { hv } else { hy })
                } else {
                    match twin {
                        Some((c, i)) => (c, i, 0),
                        None => {
                            let xcs = t.corners(&other);
                            let c = xcs as u8;
                            let i = if read && c <= r {
                                t.probe_edges(&other, xcs)
                            } else {
                                KNOWN
                            };
                            if i != KNOWN {
                                t.h48.prefetch(i);
                            }
                            twin = Some((c, i));
                            (c, i, 0)
                        }
                    }
                };
                if xc > r || xh > r {
                    continue;
                }
                let oi = if read {
                    t.probe_edges(&own, ocs)
                } else {
                    KNOWN
                };
                if oi != KNOWN {
                    t.h48.prefetch(oi);
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
        // Stage 2: read every bound first (learned bounds: a child at v needs the parent and
        // the child's same-face siblings to be at least v - 1 away).
        let mut face_max = [0u8; 6];
        let mut lb = 0u8;
        for k in kids.iter_mut().take(nk) {
            let k = unsafe { k.assume_init_mut() };
            if k.iv != KNOWN {
                k.hv = k.hv.max(t.h48.value(k.iv));
            }
            if k.iy != KNOWN {
                k.hy = k.hy.max(t.h48.value(k.iy));
            }
            let v = k.hv.max(k.hy);
            let f = (k.m / 3) as usize;
            face_max[f] = face_max[f].max(v);
            lb = lb.max(v);
        }
        if lb > left + 1 {
            return false;
        }
        // Stage 3: descend.
        for k in kids.iter().take(nk) {
            let k = unsafe { k.assume_init_ref() };
            if k.hv > r || k.hy > r || face_max[(k.m / 3) as usize] > left {
                continue;
            }
            let found = if back {
                self.back.push(k.m);
                self.dfs(
                    &k.node,
                    r,
                    last_f,
                    k.m as usize,
                    k.hv,
                    k.hy,
                    fq || k.hy == r,
                    k.hv == r,
                )
            } else {
                self.front.push(k.m);
                self.dfs(
                    &k.node,
                    r,
                    k.m as usize,
                    last_b,
                    k.hv,
                    k.hy,
                    k.hy == r,
                    bq || k.hv == r,
                )
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
