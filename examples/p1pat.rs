//! Estimate of pattern-aware phase-1 pruning: on the p1check workload, per remaining depth,
//! the nodes the current tree expands, and (on a uniform sample of them) the exact distance
//! T[f][c] to "staged, colour pair c on frame f's special axis in an allowed pattern", so the
//! share of nodes a tree pruned by T would keep (per frame, and for the min over frames).
use eggg_solver::cube::Cube;
use eggg_solver::moves::{ALLOWED_AFTER, N_MOVES};
use eggg_solver::phase1::{P1State, P1Tables};
use eggg_solver::solver::load_p1;
use eggg_solver::table::Mod3Table;
use rand::{Rng, SeedableRng};

const MAXL: usize = 24;
const K: usize = 400;

struct Pass<'a> {
    t: &'a P1Tables,
    expanded: [u64; MAXL],
    lookups: [u64; MAXL],
    seen: [u64; MAXL],
    samples: Vec<Vec<(P1State, u8)>>,
    rng: rand_xoshiro::Xoshiro256PlusPlus,
}

impl Pass<'_> {
    fn dfs(&mut self, s: &P1State, d: u8, left: u8, last: usize) {
        let l = left as usize;
        self.expanded[l] += 1;
        // Reservoir sample of the expanded nodes at this remaining depth.
        self.seen[l] += 1;
        if self.samples[l].len() < K {
            self.samples[l].push((*s, d));
        } else {
            let j = self.rng.gen_range(0..self.seen[l]);
            if (j as usize) < K {
                self.samples[l][j as usize] = (*s, d);
            }
        }
        if left == 0 {
            return;
        }
        let mut a = ALLOWED_AFTER[last];
        while a != 0 {
            let m = a.trailing_zeros() as usize;
            a &= a - 1;
            self.lookups[l] += 1;
            let c = s.apply(m, self.t);
            let dc = Mod3Table::decode(d, self.t.value(&c));
            if dc < left {
                self.dfs(&c, dc, left - 1, m);
            }
        }
    }
}

/// Which (frame, pair) goals hold: bit 3 f + c.
fn goals(t: &P1Tables, s: &P1State) -> u16 {
    let mut out = 0;
    for (f, fr) in t.frames.iter().enumerate() {
        let c = (s.centres.0[fr.rl_pos[0] as usize] >> 1) as usize;
        if fr.centres_ok(&s.centres) {
            out |= 1 << (3 * f + c);
        }
    }
    out
}

/// The goals reachable in exactly `left` more moves (canonical sequences).
fn reach(
    t: &P1Tables,
    s: &P1State,
    d: u8,
    left: u8,
    last: usize,
    want: u16,
    nodes: &mut u64,
) -> u16 {
    *nodes += 1;
    if left == 0 {
        return if d == 0 { goals(t, s) & want } else { 0 };
    }
    let mut got = 0;
    let mut a = ALLOWED_AFTER[last];
    while a != 0 {
        let m = a.trailing_zeros() as usize;
        a &= a - 1;
        let c = s.apply(m, t);
        let dc = Mod3Table::decode(d, t.value(&c));
        if dc < left {
            got |= reach(t, &c, dc, left - 1, m, want & !got, nodes);
            if got == want {
                break;
            }
        }
    }
    got
}

fn main() {
    let p1 = load_p1(std::path::Path::new("tables"), true, 4, false);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
    let mut pass = Pass {
        t: &p1,
        expanded: [0; MAXL],
        lookups: [0; MAXL],
        seen: [0; MAXL],
        samples: vec![vec![]; MAXL],
        rng: rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1),
    };
    let ncubes: usize = std::env::var("CUBES")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(5);
    let extra: u8 = std::env::var("EXTRA")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(3);
    let skip: usize = std::env::var("SKIP")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(0);
    for i in 0..ncubes + skip {
        let cube = Cube::random(&mut rng);
        if i < skip {
            continue;
        }
        let root = P1State::from_cube(&cube);
        let d0 = p1.distance(&root);
        let from: u8 = std::env::var("FROM")
            .ok()
            .map(|s| s.parse().unwrap())
            .unwrap_or(0);
        for len in d0 + from..=d0 + extra {
            pass.dfs(&root, d0, len, N_MOVES);
        }
    }
    println!(
        "left expanded lookups | kept share by frame 0 1 2, any | live pairs per frame (kept) | T-d hist (frame 0)"
    );
    let (mut tot_l, mut tot_f, mut tot_any) = (0f64, [0f64; 3], 0f64);
    let mut tot_live = 0f64;
    let mut tot_livec = 0f64;
    for l in (0..MAXL).rev() {
        if pass.expanded[l] == 0 {
            continue;
        }
        let smp = &pass.samples[l];
        let mut kept = [0usize; 3];
        let mut any = 0;
        let mut live = 0usize;
        let mut livec = 0usize;
        let mut hist = [0usize; 6];
        let mut slack = [0usize; 6];
        let mut nodes = 0u64;
        for (s, d) in smp {
            // T[f][c] capped at l + 1.
            let mut tv = [l as u8 + 1; 9];
            let mut want: u16 = 0x1ff;
            for len in *d..=l as u8 {
                let got = reach(&p1, s, *d, len, N_MOVES, want, &mut nodes);
                for i in 0..9 {
                    if got >> i & 1 == 1 {
                        tv[i] = len;
                    }
                }
                want &= !got;
                if want == 0 {
                    break;
                }
            }
            let mut k_any = false;
            for f in 0..3 {
                let n = (0..3).filter(|&c| tv[3 * f + c] as usize <= l).count();
                if n > 0 {
                    kept[f] += 1;
                    k_any = true;
                }
            }
            live += (0..9).filter(|&i| tv[i] as usize <= l).count();
            livec += (0..3)
                .filter(|&c| (0..3).any(|f| tv[3 * f + c] as usize <= l))
                .count();
            if k_any {
                any += 1;
            }
            slack[(l - *d as usize).min(5)] += 1;
            let t0 = (0..3).map(|c| tv[c]).min().unwrap();
            hist[((t0 - d) as usize).min(5)] += 1;
        }
        let n = smp.len() as f64;
        let fr: Vec<f64> = kept.iter().map(|&k| k as f64 / n).collect();
        let fa = any as f64 / n;
        // Lookups below an expanded node at this level: its children's evaluations.
        let lk = pass.lookups[l] as f64;
        // Only levels whose children need a table lookup (the last move is tested by wings).
        if l >= 2 {
            tot_l += lk;
            for f in 0..3 {
                tot_f[f] += lk * fr[f];
            }
            tot_any += lk * fa;
            tot_live += lk * live as f64 / n;
            tot_livec += lk * livec as f64 / n;
        }
        println!(
            "{l:3} {:10} {:11} | {:.3} {:.3} {:.3} {:.3} | {:.2} | {:?} slack {:?} ({} search nodes)",
            pass.expanded[l],
            pass.lookups[l],
            fr[0],
            fr[1],
            fr[2],
            fa,
            live as f64 / n,
            hist,
            slack,
            nodes
        );
    }
    println!(
        "child evaluations at levels >= 2: current {tot_l:.0}; per-frame trees {:.0} {:.0} {:.0} (sum {:.0}, {:.2}x); min-over-frames tree {:.0} ({:.2}x); lookups with live (f,c) masks {:.0} ({:.2}x)",
        tot_f[0],
        tot_f[1],
        tot_f[2],
        tot_f.iter().sum::<f64>(),
        tot_f.iter().sum::<f64>() / tot_l,
        tot_any,
        tot_any / tot_l,
        tot_live,
        tot_live / tot_l
    );
    println!(
        "per-pair trees (any frame): lookups {tot_livec:.0} ({:.2}x)",
        tot_livec / tot_l
    );
}
