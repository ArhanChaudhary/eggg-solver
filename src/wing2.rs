//! A wing lookahead for phase-1 nodes two moves from the end (method/p1-h2).
//!
//! A phase-1 endpoint's last move is a wide quarter turn (from unstaged centres only a wide
//! move stages them, and the wide half turns are phase-2 moves in every frame) or, from
//! staged centres, an outer quarter turn of the frame's special axis (the only moves that
//! keep the centres staged and are not phase-2 moves in that frame). Its wings must be
//! pairable in the frame: every dedge has one wing in the frame's low slots, an even number
//! of those wings are the frame's "high" pieces, and the wing permutation is even.
//!
//! So a node two moves from the end (moves m1, m2) can only reach an endpoint in frame f if
//! its wings are pairable with respect to the slots that m1 m2 takes to the low slots (their
//! preimage), for some (m1, m2) whose parities make the endpoint's even. That is a test of
//! the node's wing positions alone, bit-sliced over the distinct preimages: per slot, the
//! set of preimages (bits) that contain it; a dedge is split by a preimage when exactly one
//! of its wings' slots is in it. For a node at staging distance 2 (`tight`), m2 is a wide
//! quarter turn: 43 preimages per frame (12 for even wing parity, where m1 must be a wide
//! quarter turn too); with slack, m2 may also be an outer quarter turn of the special axis.
//! Nodes that fail have no endpoint below them, so they need not be expanded (the centres
//! are ignored: the test is a necessary condition).

use crate::moves::{ALLOWED_AFTER, N_MOVES, WIDE_QUARTER};
use crate::phase1::P1Tables;
use crate::simd::V32;

/// Words per frame (at most 64 * WF preimages per frame and kind).
const WF: usize = 1;

/// Outer quarter turns (U R F D L B, clockwise and anticlockwise).
const OUTER_QUARTER: u32 = {
    let mut m = 0;
    let mut f = 0;
    while f < 6 {
        m |= 0b101 << (3 * f);
        f += 1;
    }
    m
};

pub struct Wing2 {
    /// [kind (0 tight, 1 slack)][parity][slot]: the preimages (bits) that contain the slot,
    /// frame f's in words f * WF ..
    b: Box<[[[[u64; 3 * WF]; 24]; 2]; 2]>,
    /// [frame][dedge]: the frame's high piece of the dedge (2e or 2e + 1).
    high: [[u8; 12]; 3],
    /// [kind][parity][frame]: the number of preimages.
    counts: [[[usize; 3]; 2]; 2],
}

impl Wing2 {
    pub fn new(t: &P1Tables) -> Wing2 {
        let mut b = Box::new([[[[0u64; 3 * WF]; 24]; 2]; 2]);
        let mut counts = [[[0usize; 3]; 2]; 2];
        for kind in 0..2 {
            for par in 0..2 {
                for f in 0..3 {
                    let low = &t.frames[f].low_slots;
                    let last = if kind == 0 {
                        WIDE_QUARTER
                    } else {
                        WIDE_QUARTER | OUTER_QUARTER & !t.frames[f].p2_mask
                    };
                    // The distinct preimages of the low slots under m1 m2 (24-bit masks).
                    let mut pre: Vec<u32> = vec![];
                    for m1 in 0..N_MOVES {
                        let mut a = ALLOWED_AFTER[m1] & last;
                        while a != 0 {
                            let m2 = a.trailing_zeros() as usize;
                            a &= a - 1;
                            // The endpoint's wing parity: the node's, flipped by each wide
                            // quarter turn.
                            let flips = (WIDE_QUARTER >> m1 & 1) ^ (WIDE_QUARTER >> m2 & 1);
                            if par as u32 ^ flips != 0 {
                                continue;
                            }
                            let mut p = 0u32;
                            for x in 0..24 {
                                let y = t.wing_dst[m2].0[t.wing_dst[m1].0[x] as usize];
                                if low.contains(&y) {
                                    p |= 1 << x;
                                }
                            }
                            if !pre.contains(&p) {
                                pre.push(p);
                            }
                        }
                    }
                    assert!(pre.len() <= 64 * WF, "{} preimages", pre.len());
                    counts[kind][par][f] = pre.len();
                    for (j, &p) in pre.iter().enumerate() {
                        for (x, row) in b[kind][par].iter_mut().enumerate() {
                            if p >> x & 1 == 1 {
                                row[f * WF + j / 64] |= 1 << (j % 64);
                            }
                        }
                    }
                }
            }
        }
        let high = std::array::from_fn(|f| {
            std::array::from_fn(|e| {
                if t.frames[f].high_pieces >> (2 * e) & 1 == 1 {
                    2 * e as u8
                } else {
                    2 * e as u8 + 1
                }
            })
        });
        Wing2 { b, high, counts }
    }

    /// The number of preimages per kind (tight, slack), parity and frame (for reports).
    pub fn counts(&self) -> [[[usize; 3]; 2]; 2] {
        self.counts
    }

    /// Can a node with wing positions `pos` and wing parity `odd`, two moves from the end of
    /// phase 1, reach an endpoint (wings only)? `tight`: the node is at staging distance 2
    /// (its last move is then a wide quarter turn).
    #[inline(always)]
    pub fn possible(&self, pos: &V32, odd: bool, tight: bool) -> bool {
        let b = &self.b[!tight as usize][odd as usize];
        let mut ok = [!0u64; 3 * WF];
        for e in 0..12 {
            let x = &b[pos.0[2 * e] as usize];
            let y = &b[pos.0[2 * e + 1] as usize];
            for w in 0..3 * WF {
                ok[w] &= x[w] ^ y[w];
            }
        }
        if ok.iter().all(|&w| w == 0) {
            return false;
        }
        // Per frame, the parity of the high pieces in the preimage (given one wing of each
        // dedge in it) must be even.
        let mut any = 0;
        for f in 0..3 {
            let mut h = [0u64; WF];
            for &p in &self.high[f] {
                let x = &b[pos.0[p as usize] as usize];
                for w in 0..WF {
                    h[w] ^= x[f * WF + w];
                }
            }
            for w in 0..WF {
                any |= ok[f * WF + w] & !h[w];
            }
        }
        any != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::Cube;
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    /// Brute force over the move pairs.
    fn possible_ref(t: &P1Tables, pos: &V32, odd: bool, tight: bool) -> bool {
        for m1 in 0..N_MOVES {
            let p1 = t.wing_dst[m1].gather(pos);
            let mut a = ALLOWED_AFTER[m1];
            while a != 0 {
                let m2 = a.trailing_zeros() as usize;
                a &= a - 1;
                let wq = (WIDE_QUARTER >> m1 & 1) ^ (WIDE_QUARTER >> m2 & 1) == 1;
                if odd != wq {
                    continue;
                }
                let p2 = t.wing_dst[m2].gather(&p1);
                for f in 0..3 {
                    let last_ok = WIDE_QUARTER >> m2 & 1 == 1
                        || !tight
                            && OUTER_QUARTER >> m2 & 1 == 1
                            && t.frames[f].p2_mask >> m2 & 1 == 0;
                    if last_ok && t.frames[f].wings_ok(&p2) {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[test]
    fn matches_brute_force() {
        let t = P1Tables::with_table(
            crate::phase1::Staging::new(),
            crate::table::Mod3Table {
                data: crate::table::HugeBuf::zeroed(1),
            },
        );
        let w = Wing2::new(&t);
        eprintln!("preimages {:?}", w.counts());
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        let mut pass = [0; 2];
        let mut n = 0;
        let quarters: Vec<usize> = (0..N_MOVES)
            .filter(|&m| (WIDE_QUARTER | OUTER_QUARTER) >> m & 1 == 1)
            .collect();
        // Random wing states, and states two moves before a pairable one (to hit passes).
        for i in 0..40000 {
            let mut p: Vec<u8> = (0..24).collect();
            p.shuffle(&mut rng);
            let mut pos = V32::ZERO;
            pos.0[..24].copy_from_slice(&p);
            let mut odd = Cube::parity(&p) == 1;
            if i % 2 == 1 {
                // Undo two moves (the second a quarter turn) from the solved wings.
                let mut q = V32::ZERO;
                q.0[..24].copy_from_slice(&std::array::from_fn::<u8, 24, _>(|k| k as u8));
                let ms = [
                    quarters[rand::Rng::gen_range(&mut rng, 0..quarters.len())],
                    rand::Rng::gen_range(&mut rng, 0..N_MOVES),
                ];
                for &m in &ms {
                    q = t.wing_dst[crate::moves::inverse_move(m)].gather(&q);
                }
                pos = q;
                odd = (WIDE_QUARTER >> ms[0] & 1) ^ (WIDE_QUARTER >> ms[1] & 1) == 1;
            }
            for tight in [true, false] {
                let want = possible_ref(&t, &pos, odd, tight);
                assert_eq!(w.possible(&pos, odd, tight), want, "case {i} tight {tight}");
                if i % 2 == 0 {
                    pass[tight as usize] += want as usize;
                }
            }
            n += (i % 2 == 0) as usize;
        }
        eprintln!(
            "random states: pass slack {} tight {} of {n}",
            pass[0], pass[1]
        );
    }
}
