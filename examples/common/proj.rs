//! Exact distances in projections of the phase-2 state, by IDA* with the joint table's two
//! views as bounds: what the large joint tables (src/p2big.rs) hold, computed without them.
//! Shared by examples p2deep and p2bigcheck.
use eggg_solver::moves::{ALLOWED_AFTER, N_MOVES, PHASE2_MASK};
use eggg_solver::phase2::{P2_MOVES, P2Tables};
use eggg_solver::table::Mod3Table;

/// The projections the oracle can use: 0 = (U/D, F/B, tau, PLL), 1 = (U/D, R/L, tau, PLL),
/// 2 = (F/B, R/L, tau, PLL). Their goals ignore which way round each solved pattern is.
pub struct Proj<'a> {
    pub t: &'a P2Tables,
    /// R/L pattern ranks of the solved patterns.
    rl_goal: [bool; 12],
    /// Distance of each R/L pattern to a solved one.
    rl_dist: [u8; 12],
}

impl<'a> Proj<'a> {
    pub fn new(t: &'a P2Tables) -> Self {
        let mut rl_goal = [false; 12];
        for cc in 0..t.centre_dist.len() {
            if t.centre_dist[cc] == 0 {
                rl_goal[cc / 2 % 12] = true;
            }
        }
        let mut rl_dist = [u8::MAX; 12];
        let mut fr: Vec<usize> = (0..12).filter(|&r| rl_goal[r]).collect();
        for &r in &fr {
            rl_dist[r] = 0;
        }
        let mut d = 0;
        while !fr.is_empty() {
            d += 1;
            let mut next = vec![];
            for &r in &fr {
                for i in 0..17 {
                    let r2 = (t.centre_move[r * 2][i] / 2 % 12) as usize;
                    if rl_dist[r2] == u8::MAX {
                        rl_dist[r2] = d;
                        next.push(r2);
                    }
                }
            }
            fr = next;
        }
        Proj {
            t,
            rl_goal,
            rl_dist,
        }
    }

    fn goal(&self, kind: u8, cc: u32, dj: u8, df: u8) -> bool {
        let rl = (cc / 2 % 12) as usize;
        match kind {
            0 => dj == 0 && df == 0,
            1 => dj == 0 && self.rl_goal[rl],
            _ => df == 0 && self.rl_goal[rl],
        }
    }

    fn bound(&self, kind: u8, cc: u32, dj: u8, df: u8) -> u8 {
        let rl = self.rl_dist[(cc / 2 % 12) as usize];
        match kind {
            0 => dj.max(df),
            1 => dj.max(rl),
            _ => df.max(rl),
        }
    }

    /// Within `left` moves of the projection's goal?
    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        kind: u8,
        cc: u32,
        tau: &[u8; 16],
        dj: u8,
        df: u8,
        left: u8,
        last: usize,
    ) -> bool {
        if self.goal(kind, cc, dj, df) {
            return true;
        }
        if left == 0 {
            return false;
        }
        let t = self.t;
        let j = t.joint.as_ref().unwrap();
        let allowed = ALLOWED_AFTER[last] & PHASE2_MASK;
        for (i, &m) in P2_MOVES.iter().enumerate() {
            if allowed >> m & 1 == 0 {
                continue;
            }
            let c2 = t.centre_move[cc as usize][i];
            let t2 = t.tau_move(tau, i);
            let d2 = Mod3Table::decode(dj, j.get(j.index(c2 / 1680, &t2, c2 & 1)));
            let f2 = Mod3Table::decode(df, j.get(j.index_fb(c2, &t2)));
            if self.bound(kind, c2, d2, f2) >= left {
                continue;
            }
            if self.search(kind, c2, &t2, d2, f2, left - 1, m as usize) {
                return true;
            }
        }
        false
    }

    /// Exact distance in projection `kind` (memoised on the projected state).
    pub fn dist(&self, kind: u8, cc: u32, tau: &[u8; 16], dj: u8, df: u8, memo: &mut Memo) -> u8 {
        let (ud, fb, rl, pll) = (cc / 1680, cc / 24 % 70, cc / 2 % 12, cc & 1);
        let key_cc = match kind {
            0 => ((ud * 70 + fb) * 12) * 2 + pll,
            1 => (ud * 70 * 12 + rl) * 2 + pll,
            _ => (fb * 12 + rl) * 2 + pll,
        };
        let key = (kind, key_cc, *tau);
        if let Some(&d) = memo.get(&key) {
            return d;
        }
        let mut b = self.bound(kind, cc, dj, df);
        while !self.search(kind, cc, tau, dj, df, b, N_MOVES) {
            b += 1;
        }
        memo.insert(key, b);
        b
    }
}

pub type Memo = std::collections::HashMap<(u8, u32, [u8; 16]), u8>;
