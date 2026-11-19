//! Phase-1 / phase-2 interaction analysis (method/p1-h2).
//!
//! For random cubes and phase-1 lengths LENS, a sample of the phase-1 jobs (3-move prefixes,
//! as the driver makes them; 1 in FRAC, the same prefixes at every length) is searched with a
//! depth-first mirror of `Search1::dfs_eq` (reduced move masks, parity viability), and every
//! endpoint candidate (endpoint x frame) gets its exact phase-2 bound h2 = max(centre
//! distance, joint distance). Per node of the tree (by moves left k), the smallest h2 of the
//! candidates below it is recorded, with a few node features; per job, the h2 histogram.
//! Env: CUBES (8), SEED (6060), LENS (14,15,16), FRAC (64), THREADS (8), CHECK (0), JOBS
//! (file for per-job records), TABLES (tables).
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, MOVE_PERMS, N_MOVES};
use eggg_solver::phase1::{ALL_PAIRS, P1State, P1Tables, Search1, trailing_run};
use eggg_solver::phase2::{P2Tables, standardise};
use eggg_solver::solver::{load_mask, load_p1, load_p2};
use eggg_solver::table::{Mod3Table, UNSEEN};
use rand::SeedableRng;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Moves left tracked: 0..KM (larger k lumped into KM - 1).
const KM: usize = 10;
/// h2 values 0..NONE-1; NONE: no candidate below.
const NH: usize = 24;
const NONE: u8 = (NH - 1) as u8;

#[derive(Clone)]
struct Acc {
    /// [len][k][min h2 below]: nodes.
    nodes: Vec<[[u64; NH]; KM]>,
    /// [len][cd][dt]: candidates.
    ep: Vec<[[u64; NH]; NH]>,
    /// [len][k][feature][min h2 below]: paired dedges (both wings in one edge position).
    f_pair: Vec<[[[u64; NH]; 13]; KM]>,
    /// [len][k][feature][min h2 below]: fewest bad pairs (both wings low or both high) over
    /// the frames.
    f_bad: Vec<[[[u64; NH]; 13]; KM]>,
    /// [len][k][wide quarter turns among the last k moves... ] unused
    /// [len][h2 of the candidate][frame]
    ep_frame: Vec<[[u64; 3]; NH]>,
    /// [len][wing-only pairable within 2 moves][min h2 below], sampled k = 2 nodes.
    w2: Vec<[[u64; NH]; 2]>,
    /// The same restricted as at a tight k = 2 node (distance 2): exactly two moves, the
    /// last a wide quarter turn, ending with even wing parity: [len][pass][min h2 below].
    w2r: Vec<[[u64; NH]; 2]>,
    /// Sibling structure: [len][k][min h2 below][min h2 of the other children's subtrees
    /// (NONE if no other child has a candidate)] for nodes with a candidate below.
    sib: Vec<[[[u64; NH]; NH]; KM]>,
}

impl Acc {
    fn new(n: usize) -> Acc {
        Acc {
            nodes: vec![[[0; NH]; KM]; n],
            ep: vec![[[0; NH]; NH]; n],
            f_pair: vec![[[[0; NH]; 13]; KM]; n],
            f_bad: vec![[[[0; NH]; 13]; KM]; n],
            ep_frame: vec![[[0; 3]; NH]; n],
            sib: vec![[[[0; NH]; NH]; KM]; n],
            w2: vec![[[0; NH]; 2]; n],
            w2r: vec![[[0; NH]; 2]; n],
        }
    }
    fn add(&mut self, o: &Acc) {
        for l in 0..self.nodes.len() {
            for k in 0..KM {
                for h in 0..NH {
                    self.nodes[l][k][h] += o.nodes[l][k][h];
                    for f in 0..13 {
                        self.f_pair[l][k][f][h] += o.f_pair[l][k][f][h];
                        self.f_bad[l][k][f][h] += o.f_bad[l][k][f][h];
                    }
                    for g in 0..NH {
                        self.sib[l][k][h][g] += o.sib[l][k][h][g];
                    }
                }
            }
            for w in 0..2 {
                for h in 0..NH {
                    self.w2[l][w][h] += o.w2[l][w][h];
                    self.w2r[l][w][h] += o.w2r[l][w][h];
                }
            }
            for a in 0..NH {
                for b in 0..NH {
                    self.ep[l][a][b] += o.ep[l][a][b];
                }
                for f in 0..3 {
                    self.ep_frame[l][a][f] += o.ep_frame[l][a][f];
                }
            }
        }
    }
}

struct Ctx<'a> {
    p1: &'a P1Tables,
    p2: &'a P2Tables,
    /// Per frame, per slot: low?
    low: [[bool; 24]; 3],
}

struct Walk<'a, 'c> {
    c: &'c Ctx<'a>,
    cube: Cube,
    path: Vec<u8>,
    li: usize,
    acc: &'c mut Acc,
    /// This job's candidates by h2.
    job_h: [u32; NH],
    visits: u64,
}

impl Walk<'_, '_> {
    /// The candidates of the endpoint at the end of `path` (frames `frames`): their smallest h2.
    fn endpoint(&mut self, frames: u8) -> u8 {
        let mut c = self.cube;
        for &m in &self.path {
            c = c.apply_perm(&MOVE_PERMS[m as usize]);
        }
        let p2 = self.c.p2;
        let mut best = NONE;
        for frame in 0..3 {
            if frames >> frame & 1 == 0 {
                continue;
            }
            let x = standardise(&c, frame);
            let centre = p2.centre_coord_of(&x);
            let cd = p2.centre_dist[centre as usize];
            if cd == UNSEEN {
                continue;
            }
            let tau = p2.tau_of(&x);
            let dt = p2
                .joint
                .as_ref()
                .unwrap()
                .distance(centre / 1680, &tau, centre & 1);
            let h = cd.max(dt).min(NONE - 1);
            self.acc.ep[self.li][(cd as usize).min(NH - 1)][(dt as usize).min(NH - 1)] += 1;
            self.acc.ep_frame[self.li][h as usize][frame] += 1;
            self.job_h[h as usize] += 1;
            best = best.min(h);
        }
        best
    }

    fn features(&self, s: &P1State) -> (usize, usize) {
        let pos = &s.pos.0;
        let mut paired = 0;
        let mut bad = [0usize; 3];
        for e in 0..12 {
            let (a, b) = (pos[2 * e] as usize, pos[2 * e + 1] as usize);
            if a / 2 == b / 2 {
                paired += 1;
            }
            for f in 0..3 {
                if self.c.low[f][a] == self.c.low[f][b] {
                    bad[f] += 1;
                }
            }
        }
        (paired, *bad.iter().min().unwrap())
    }

    /// Mirror of `Search1::dfs_eq` (all pairs, reduced masks): the smallest h2 of the
    /// candidates below `s` (NONE if none).
    fn dfs(&mut self, s: &P1State, d: u8, left: u8, last: usize, r: (u64, u8)) -> u8 {
        let t = self.c.p1;
        if left == 0 {
            if d != 0 {
                return NONE;
            }
            let frames = t.endpoint_frames(s, trailing_run(&self.path));
            if frames == 0 {
                return NONE;
            }
            return self.endpoint(frames);
        }
        self.visits += 1;
        let k = (left as usize).min(KM - 1);
        let mt = t.mask.as_ref().unwrap();
        let (fp, fb) = self.features(s);
        if left >= 2 && d == left && !mt.viable(r.0, s.parity) {
            self.record(k, fp, fb, NONE, NONE);
            return NONE;
        }
        let (dec, eq) = mt.get_dec_eq(r.0, r.1);
        let allowed = ALLOWED_AFTER[last];
        let (dec, eq) = (dec & allowed, eq & allowed);
        let visit = if d + 1 < left { allowed } else { dec | eq };
        let mut a = visit;
        let mut hs: [u8; N_MOVES] = [NONE; N_MOVES];
        let mut best = NONE;
        while a != 0 {
            let m = a.trailing_zeros() as usize;
            a &= a - 1;
            let dc = d + 1 - 2 * (dec >> m & 1) as u8 - (eq >> m & 1) as u8;
            if dc > left - 1 {
                continue;
            }
            let c = s.apply(m, t);
            let rc = if left >= 2 {
                mt.resolve(t, mt.pre_index(t, &c, ALL_PAIRS), &c, ALL_PAIRS)
            } else {
                (0, 0)
            };
            self.path.push(m as u8);
            let h = self.dfs(&c, dc, left - 1, m, rc);
            self.path.pop();
            hs[m] = h;
            best = best.min(h);
        }
        // The best child's subtree vs the best of the others.
        let mut other = NONE;
        if best != NONE {
            let mut seen_best = false;
            for &h in &hs {
                if h == best && !seen_best {
                    seen_best = true;
                } else {
                    other = other.min(h);
                }
            }
        }
        self.record(k, fp, fb, best, other);
        if left == 2 && self.visits.is_multiple_of(16) {
            let w = self.wing2(s) as usize;
            self.acc.w2[self.li][w][best as usize] += 1;
            if d == 2 {
                let w = self.wing2r(s) as usize;
                self.acc.w2r[self.li][w][best as usize] += 1;
            }
        }
        best
    }

    /// [`Walk::wing2`] with exactly two moves, the last a wide quarter turn.
    fn wing2r(&self, s: &P1State) -> bool {
        use eggg_solver::moves::WIDE_QUARTER;
        let t = self.c.p1;
        for m1 in 0..N_MOVES {
            let o1 = s.parity ^ (WIDE_QUARTER >> m1 & 1 == 1);
            if !o1 {
                continue;
            }
            let p1 = t.wing_dst[m1].gather(&s.pos);
            let mut a = ALLOWED_AFTER[m1] & WIDE_QUARTER;
            while a != 0 {
                let m2 = a.trailing_zeros() as usize;
                a &= a - 1;
                let p2 = t.wing_dst[m2].gather(&p1);
                if t.frames.iter().any(|f| f.wings_ok(&p2)) {
                    return true;
                }
            }
        }
        false
    }

    /// Can the wings be made pairable (even parity) in some frame by at most 2 moves (any)?
    fn wing2(&self, s: &P1State) -> bool {
        let t = self.c.p1;
        let ok = |pos: &eggg_solver::simd::V32, odd: bool| {
            !odd && t.frames.iter().any(|f| f.wings_ok(pos))
        };
        if ok(&s.pos, s.parity) {
            return true;
        }
        for m1 in 0..N_MOVES {
            let p1 = t.wing_dst[m1].gather(&s.pos);
            let o1 = s.parity ^ (eggg_solver::moves::WIDE_QUARTER >> m1 & 1 == 1);
            if ok(&p1, o1) {
                return true;
            }
            let mut a = ALLOWED_AFTER[m1];
            while a != 0 {
                let m2 = a.trailing_zeros() as usize;
                a &= a - 1;
                let p2 = t.wing_dst[m2].gather(&p1);
                if ok(&p2, o1 ^ (eggg_solver::moves::WIDE_QUARTER >> m2 & 1 == 1)) {
                    return true;
                }
            }
        }
        false
    }

    fn record(&mut self, k: usize, fp: usize, fb: usize, h: u8, other: u8) {
        let (li, h) = (self.li, h as usize);
        self.acc.nodes[li][k][h] += 1;
        self.acc.f_pair[li][k][fp][h] += 1;
        self.acc.f_bad[li][k][fb.min(12)][h] += 1;
        self.acc.sib[li][k][h][other as usize] += 1;
    }
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let cubes: usize = env("CUBES", 8);
    let seed: u64 = env("SEED", 6060);
    let frac: u64 = env("FRAC", 64);
    let threads: usize = env("THREADS", 8);
    let check: usize = env("CHECK", 0);
    let lens: Vec<u8> = std::env::var("LENS")
        .unwrap_or("14,15,16".into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let dir = std::path::PathBuf::from(std::env::var("TABLES").unwrap_or("tables".into()));
    let mut p1 = load_p1(&dir, false, threads, true);
    p1.table.data.collapse();
    load_mask(&mut p1, &dir, false, true);
    p1.mask.as_ref().unwrap().collapse();
    let p2 = load_p2(&dir, true, threads, true);
    p2.joint.as_ref().unwrap().table.collapse();
    let low = std::array::from_fn(|f| {
        let mut l = [false; 24];
        for &s in &p1.frames[f].low_slots {
            l[s as usize] = true;
        }
        l
    });
    let ctx = Ctx {
        p1: &p1,
        p2: &p2,
        low,
    };
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    // Work items: (cube, len index, prefix).
    let mut items = vec![];
    let mut roots = vec![];
    for ci in 0..cubes {
        let cube = Cube::random(&mut rng);
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        eprintln!("cube {ci}: staging distance {d0}");
        roots.push((cube, root, d0));
        // 3-move canonical prefixes, sampled by a hash of (cube, prefix).
        for m0 in 0..N_MOVES {
            for m1 in 0..N_MOVES {
                if ALLOWED_AFTER[m0] >> m1 & 1 == 0 {
                    continue;
                }
                for m2 in 0..N_MOVES {
                    if ALLOWED_AFTER[m1] >> m2 & 1 == 0 {
                        continue;
                    }
                    let pid = (m0 * 729 + m1 * 27 + m2) as u64;
                    let h = (pid ^ (ci as u64) << 20).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 40;
                    if !h.is_multiple_of(frac) {
                        continue;
                    }
                    for (li, &len) in lens.iter().enumerate() {
                        items.push((ci, li, len, [m0 as u8, m1 as u8, m2 as u8], pid));
                    }
                }
            }
        }
    }
    // Jobs in a pseudo-random order (cubes and prefixes interleaved), each job's lengths
    // together, longest first: a run stopped by the time budget has the same jobs at every
    // length.
    items.sort_by_key(|x| {
        let h = (x.4 ^ (x.0 as u64) << 20).wrapping_mul(0x2545_f491_4f6c_dd1d);
        (h, std::cmp::Reverse(x.2))
    });
    let budget: f64 = env("BUDGET", 1e9);
    eprintln!("{} work items", items.len());
    let next = AtomicUsize::new(0);
    let total = Mutex::new(Acc::new(lens.len()));
    let jobs = Mutex::new(vec![]);
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut acc = Acc::new(lens.len());
                let mut myjobs = vec![];
                loop {
                    if t0.elapsed().as_secs_f64() > budget {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    if i.is_multiple_of(200) {
                        eprintln!(
                            "item {i}/{} at {:.0} s",
                            items.len(),
                            t0.elapsed().as_secs_f64()
                        );
                    }
                    let (ci, li, len, prefix, pid) = items[i];
                    let (cube, root, d0) = roots[ci];
                    // As `Run::level_jobs`: the prefix must keep d <= moves left.
                    let mut st = root;
                    let mut d = d0;
                    let mut ok = true;
                    for (j, &m) in prefix.iter().enumerate() {
                        st = st.apply(m as usize, &p1);
                        d = Mod3Table::decode(d, p1.value(&st));
                        if d as usize > len as usize - j - 1 {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    let mt = p1.mask.as_ref().unwrap();
                    let r = mt.resolve(&p1, mt.pre_index(&p1, &st, ALL_PAIRS), &st, ALL_PAIRS);
                    let mut w = Walk {
                        c: &ctx,
                        cube,
                        path: prefix.to_vec(),
                        li,
                        acc: &mut acc,
                        job_h: [0; NH],
                        visits: 0,
                    };
                    let left = len - 3;
                    w.dfs(&st, d, left, prefix[2] as usize, r);
                    let (job_h, visits) = (w.job_h, w.visits);
                    if i < check {
                        let mut n = 0usize;
                        let mut s = Search1::new(&p1, |_p: &[u8], f| {
                            n += f.count_ones() as usize;
                            true
                        });
                        s.path.extend_from_slice(&prefix);
                        s.dfs(&st, d, left, prefix[2] as usize);
                        drop(s);
                        let mine: u32 = job_h.iter().sum();
                        eprintln!("check item {i}: search1 {n} candidates, mine {mine} (+ unseen)");
                    }
                    myjobs.push((ci, len, pid, visits, job_h));
                }
                total.lock().unwrap().add(&acc);
                jobs.lock().unwrap().extend(myjobs);
            });
        }
    });
    eprintln!(
        "done in {:.1} s ({} of {} items started)",
        t0.elapsed().as_secs_f64(),
        next.load(Ordering::Relaxed).min(items.len()),
        items.len()
    );
    let acc = total.into_inner().unwrap();
    let jobs = jobs.into_inner().unwrap();
    if let Ok(path) = std::env::var("JOBS") {
        let mut f = std::fs::File::create(path).unwrap();
        for (ci, len, pid, visits, h) in &jobs {
            let hs: Vec<String> = h.iter().map(|x| x.to_string()).collect();
            writeln!(f, "{ci} {len} {pid} {visits} {}", hs.join(" ")).unwrap();
        }
    }
    report(&acc, &lens);
}

fn report(acc: &Acc, lens: &[u8]) {
    for (li, &len) in lens.iter().enumerate() {
        println!("=== L1 = {len} ===");
        let ep = &acc.ep[li];
        let n: u64 = ep.iter().flatten().sum();
        let mut h2 = [0u64; NH];
        let mut cd_bind = 0u64;
        let mut cd_h = [0u64; NH];
        let mut dt_h = [0u64; NH];
        for cd in 0..NH {
            for dt in 0..NH {
                let x = ep[cd][dt];
                h2[cd.max(dt)] += x;
                cd_h[cd] += x;
                dt_h[dt] += x;
                if cd > dt {
                    cd_bind += x;
                }
            }
        }
        println!(
            "candidates {n}; cd > dt in {:.2}%",
            100.0 * cd_bind as f64 / n as f64
        );
        println!("h2 hist: {}", fmt_hist(&h2, n));
        println!("cd hist: {}", fmt_hist(&cd_h, n));
        println!("dt hist: {}", fmt_hist(&dt_h, n));
        for limit in [24usize, 25, 26, 27] {
            let room = limit as i64 - len as i64;
            let kept: u64 = (0..NH).filter(|&h| (h as i64) <= room).map(|h| h2[h]).sum();
            let mut exc = String::new();
            for e in 1..=6 {
                let h = room + e;
                let c = if h >= 0 && (h as usize) < NH {
                    h2[h as usize]
                } else {
                    0
                };
                exc += &format!(" +{e}:{:.1}%", 100.0 * c as f64 / (n - kept).max(1) as f64);
            }
            // Candidates dropped only by the centre distance (dt within room).
            let mut by_cd = 0;
            for cd in 0..NH {
                for dt in 0..NH {
                    if cd as i64 > room && dt as i64 <= room {
                        by_cd += ep[cd][dt];
                    }
                }
            }
            println!(
                "limit {limit} (h2 <= {room}): kept {:.2}%; dropped excess{exc}; dropped by cd alone {:.2}% of dropped",
                100.0 * kept as f64 / n as f64,
                100.0 * by_cd as f64 / (n - kept).max(1) as f64
            );
        }
        let fr = &acc.ep_frame[li];
        for limit in [25usize, 26] {
            let room = limit as i64 - len as i64;
            let mut s = String::new();
            for f in 0..3 {
                let tot: u64 = (0..NH).map(|h| fr[h][f]).sum();
                let kept: u64 = (0..NH)
                    .filter(|&h| h as i64 <= room)
                    .map(|h| fr[h][f])
                    .sum();
                s += &format!(
                    " frame {f}: {tot} kept {:.2}%",
                    100.0 * kept as f64 / tot.max(1) as f64
                );
            }
            println!("limit {limit}:{s}");
        }
        // Nodes by moves left.
        let nodes = &acc.nodes[li];
        let all: u64 = nodes.iter().flatten().sum();
        println!(
            "nodes visited (k >= 1): {all}; candidates per node {:.2e}",
            n as f64 / all as f64
        );
        for limit in [25usize, 26] {
            let room = limit as i64 - len as i64;
            println!(
                "  limit {limit}: k, nodes, share of all, with candidate, with kept candidate, kept-only-fraction of productive"
            );
            let mut none_tot = 0u64;
            let mut dropped_tot = 0u64;
            for k in 1..KM {
                let row = &nodes[k];
                let tot: u64 = row.iter().sum();
                if tot == 0 {
                    continue;
                }
                let none = row[NONE as usize];
                let kept: u64 = (0..NH - 1)
                    .filter(|&h| h as i64 <= room)
                    .map(|h| row[h])
                    .sum();
                none_tot += none;
                dropped_tot += tot - none - kept;
                println!(
                    "  k={k}{}: {tot} ({:.1}%), cand {:.3}%, kept {:.4}%, kept/productive {:.1}%",
                    if k == KM - 1 { "+" } else { "" },
                    100.0 * tot as f64 / all as f64,
                    100.0 * (tot - none) as f64 / tot as f64,
                    100.0 * kept as f64 / tot as f64,
                    100.0 * kept as f64 / (tot - none).max(1) as f64
                );
            }
            println!(
                "  nodes with no candidate below {:.2}%, only dropped candidates {:.2}%",
                100.0 * none_tot as f64 / all as f64,
                100.0 * dropped_tot as f64 / all as f64
            );
        }
        // Features: per k in 1..=4, per value, nodes / productive / kept@26.
        let room = 26i64 - len as i64;
        for (name, f) in [("paired", &acc.f_pair[li]), ("minbad", &acc.f_bad[li])] {
            for k in 1..=4 {
                let mut s = String::new();
                for v in 0..13 {
                    let row = &f[k][v];
                    let tot: u64 = row.iter().sum();
                    if tot == 0 {
                        continue;
                    }
                    let none = row[NONE as usize];
                    let kept: u64 = (0..NH - 1)
                        .filter(|&h| h as i64 <= room)
                        .map(|h| row[h])
                        .sum();
                    // Mean min h2 over productive nodes.
                    let (sh, sn) = (0..NH - 1)
                        .fold((0u64, 0u64), |a, h| (a.0 + h as u64 * row[h], a.1 + row[h]));
                    s += &format!(
                        " [{v}: n {tot} cand {:.3}% kept {:.4}% meanh2 {:.2}]",
                        100.0 * (tot - none) as f64 / tot as f64,
                        100.0 * kept as f64 / tot as f64,
                        sh as f64 / sn.max(1) as f64
                    );
                }
                println!("  {name} k={k}:{s}");
            }
        }
        {
            let w2 = &acc.w2[li];
            let tot: u64 = w2.iter().flatten().sum();
            let pass: u64 = w2[1].iter().sum();
            let prod_fail: u64 = (0..NH - 1).map(|h| w2[0][h]).sum();
            let prod_pass: u64 = (0..NH - 1).map(|h| w2[1][h]).sum();
            println!(
                "  k=2 sampled {tot}: wing-only pairable within 2 moves {:.2}%; productive among pass {:.3}%, among fail {prod_fail}",
                100.0 * pass as f64 / tot.max(1) as f64,
                100.0 * prod_pass as f64 / pass.max(1) as f64
            );
        }
        {
            let w2 = &acc.w2r[li];
            let tot: u64 = w2.iter().flatten().sum();
            let pass: u64 = w2[1].iter().sum();
            let prod_fail: u64 = (0..NH - 1).map(|h| w2[0][h]).sum();
            let prod_pass: u64 = (0..NH - 1).map(|h| w2[1][h]).sum();
            println!(
                "  k=2 tight sampled {tot}: restricted (m1, wide quarter) pairable {:.2}%; productive among pass {:.3}%, among fail {prod_fail}",
                100.0 * pass as f64 / tot.max(1) as f64,
                100.0 * prod_pass as f64 / pass.max(1) as f64
            );
        }
        // Siblings: for productive nodes at k, how often the best other child's subtree is
        // within 0 / 1 / 2 of the best.
        for k in 1..=4 {
            let sib = &acc.sib[li][k];
            let mut s = String::new();
            let mut corr = [0u64; 5];
            let mut prod = 0;
            for h in 0..NH - 1 {
                for g in 0..NH {
                    let x = sib[h][g];
                    prod += x;
                    if g == NONE as usize {
                        corr[4] += x;
                    } else {
                        corr[(g - h).min(3)] += x;
                    }
                }
            }
            for (i, c) in corr.iter().enumerate() {
                s += &format!(
                    " {}:{:.1}%",
                    ["=", "+1", "+2", "+3..", "single"][i],
                    100.0 * *c as f64 / prod.max(1) as f64
                );
            }
            println!("  siblings k={k} (productive {prod}):{s}");
        }
    }
}

fn fmt_hist(h: &[u64], n: u64) -> String {
    let mut s = String::new();
    for (i, &x) in h.iter().enumerate() {
        if x > 0 {
            s += &format!(" {i}:{:.2}%", 100.0 * x as f64 / n as f64);
        }
    }
    s
}
