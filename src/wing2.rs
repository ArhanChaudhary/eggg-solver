//! A wing lookahead for phase-1 nodes two moves from the end.
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
//! preimage), for some (m1, m2) whose wide quarter turns make the endpoint's wing parity
//! even. That is a test of the node's wing positions alone, bit-sliced over the distinct
//! preimages: per slot, the set of preimages (bits) that contain it; a dedge is split by a
//! preimage when exactly one of its wings' slots is in it, and the high wings in it are then
//! one per dedge. For a node at staging distance 2 ("tight"), m2 is a wide quarter turn: 43
//! preimages per frame (12 for even wing parity, where m1 must be a wide quarter turn too);
//! with slack, m2 may also be an outer quarter turn of the special axis (51 / 42). Each
//! frame's preimages fit one 64-bit word.
//!
//! The test ignores the centres (and the canonical-sequence and trailing-run rules), so it is
//! a necessary condition: a node that fails it has no endpoint below it and is not searched.
//! On random wing states 5% of tight and 9% of slack nodes pass; in the phase-1 search this
//! removes ~90% of the nodes two moves from the end and of the last-move tests below them.

use crate::moves::{ALLOWED_AFTER, N_MOVES, WIDE_QUARTER};
use crate::phase1::Frame;
use crate::simd::V32;

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

/// Per frame f (word f; word 3 unused), a set of preimages (bits).
#[derive(Clone, Copy, Default)]
#[repr(C, align(32))]
struct Row([u64; 4]);

pub struct Wing2 {
    /// [kind (0 tight, 1 slack)][wing parity][slot]: the preimages that contain the slot.
    b: [[[Row; 24]; 2]; 2],
    /// Per frame (word), all ones if an odd number of dedges have their second wing (2e + 1)
    /// as the frame's high piece.
    high_odd: Row,
    /// Tight nodes of even wing parity have only 12 preimages per frame: per slot, those of
    /// the three frames in one word (bits 16 f..), and `high_odd` likewise.
    te: [u64; 24],
    te_high_odd: u64,
}

impl Wing2 {
    /// The lookahead for the three frames, `wing_dst[m]` giving the destination slot of each
    /// slot under move m.
    pub fn new(frames: &[Frame; 3], wing_dst: &[V32; N_MOVES]) -> Wing2 {
        let mut b = [[[Row::default(); 24]; 2]; 2];
        for (kind, bk) in b.iter_mut().enumerate() {
            for (par, bp) in bk.iter_mut().enumerate() {
                for (f, fr) in frames.iter().enumerate() {
                    for (j, p) in Self::preimages(fr, wing_dst, kind == 0, par == 1)
                        .into_iter()
                        .enumerate()
                    {
                        for (x, row) in bp.iter_mut().enumerate() {
                            if p >> x & 1 == 1 {
                                row.0[f] |= 1 << j;
                            }
                        }
                    }
                }
            }
        }
        let mut high_odd = Row::default();
        for (f, fr) in frames.iter().enumerate() {
            // Each dedge has one high piece; count those that are the second wing.
            if (fr.high_pieces & 0xaa_aaaa).count_ones() % 2 == 1 {
                high_odd.0[f] = !0;
            }
        }
        let mut te = [0u64; 24];
        let mut te_high_odd = 0u64;
        for (f, fr) in frames.iter().enumerate() {
            let pre = Self::preimages(fr, wing_dst, true, false);
            assert!(pre.len() <= 16);
            for (j, p) in pre.iter().enumerate() {
                for (x, w) in te.iter_mut().enumerate() {
                    if p >> x & 1 == 1 {
                        *w |= 1 << (16 * f + j);
                    }
                }
            }
            if high_odd.0[f] != 0 {
                te_high_odd |= 0xffff << (16 * f);
            }
        }
        Wing2 {
            b,
            high_odd,
            te,
            te_high_odd,
        }
    }

    /// The distinct preimages (24-bit slot masks) of frame `fr`'s low slots under the move
    /// pairs (m1, m2) that can end phase 1 from a node with wing parity `odd`, tight or not.
    fn preimages(fr: &Frame, wing_dst: &[V32; N_MOVES], tight: bool, odd: bool) -> Vec<u32> {
        let last = if tight {
            WIDE_QUARTER
        } else {
            WIDE_QUARTER | OUTER_QUARTER & !fr.p2_mask
        };
        let mut pre: Vec<u32> = vec![];
        for m1 in 0..N_MOVES {
            let mut a = ALLOWED_AFTER[m1] & last;
            while a != 0 {
                let m2 = a.trailing_zeros() as usize;
                a &= a - 1;
                // The endpoint's wing parity: the node's, flipped by each wide quarter turn.
                let flips = (WIDE_QUARTER >> m1 & 1) ^ (WIDE_QUARTER >> m2 & 1);
                if odd as u32 ^ flips != 0 {
                    continue;
                }
                let mut p = 0u32;
                for x in 0..24 {
                    let y = wing_dst[m2].0[wing_dst[m1].0[x] as usize];
                    if fr.low_slots.contains(&y) {
                        p |= 1 << x;
                    }
                }
                if !pre.contains(&p) {
                    pre.push(p);
                }
            }
        }
        assert!(pre.len() <= 64, "{} preimages", pre.len());
        pre
    }

    /// Can a node with wing positions `pos` (the slot of each wing) and wing parity `odd`, two
    /// moves from the end of phase 1, reach an endpoint as far as its wings tell? `tight`: the
    /// node is at staging distance 2 (its last move is then a wide quarter turn).
    #[inline(always)]
    pub fn possible(&self, pos: &V32, odd: bool, tight: bool) -> bool {
        // (The wing slots are < 24: no bounds checks on the rows, which cost as much as the
        // test on aarch64.)
        debug_assert!(pos.0[..24].iter().all(|&x| x < 24));
        if tight && !odd {
            // The 36 preimages of the three frames in one word.
            let mut ok = !0u64;
            let mut h = 0u64;
            for e in 0..12 {
                // SAFETY: wing slots are < 24.
                let (x, y) = unsafe {
                    (
                        *self.te.get_unchecked(pos.0[2 * e] as usize),
                        *self.te.get_unchecked(pos.0[2 * e + 1] as usize),
                    )
                };
                ok &= x ^ y;
                h ^= x;
            }
            return ok & !(h ^ self.te_high_odd) != 0;
        }
        let b = &self.b[!tight as usize][odd as usize];
        // Per preimage (bit): does it split every dedge (ok), and the parity h of the number
        // of dedges whose first wing (2e) is in it. Where every dedge is split, a dedge's high
        // piece is in the preimage exactly when its first wing is, flipped if the high piece
        // is the second wing; so the high pieces in it number h plus the frame's count of
        // second-wing high pieces, mod 2, which must be even.
        let mut ok = [!0u64; 3];
        let mut h = [0u64; 3];
        for e in 0..12 {
            // SAFETY: wing slots are < 24.
            let (x, y) = unsafe {
                (
                    &b.get_unchecked(pos.0[2 * e] as usize).0,
                    &b.get_unchecked(pos.0[2 * e + 1] as usize).0,
                )
            };
            for w in 0..3 {
                ok[w] &= x[w] ^ y[w];
                h[w] ^= x[w];
            }
        }
        let mut any = 0;
        for w in 0..3 {
            any |= ok[w] & !(h[w] ^ self.high_odd.0[w]);
        }
        any != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cube::Cube;
    use crate::phase1::{P1Tables, Staging};
    use crate::table::Mod3Table;
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    /// [`Wing2::possible`] by brute force over the move pairs.
    fn possible_ref(t: &P1Tables, pos: &V32, odd: bool, tight: bool) -> bool {
        for m1 in 0..N_MOVES {
            let p1 = t.wing_dst[m1].gather(pos);
            let mut a = ALLOWED_AFTER[m1];
            while a != 0 {
                let m2 = a.trailing_zeros() as usize;
                a &= a - 1;
                let flips = (WIDE_QUARTER >> m1 & 1) ^ (WIDE_QUARTER >> m2 & 1) == 1;
                if odd != flips {
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
    fn preimage_counts() {
        let t = P1Tables::with_table(Staging::new(), Mod3Table::from_depths(&[0]));
        for (tight, odd, want) in [
            (true, false, 12),
            (true, true, 43),
            (false, false, 42),
            (false, true, 51),
        ] {
            for fr in &t.frames {
                assert_eq!(Wing2::preimages(fr, &t.wing_dst, tight, odd).len(), want);
            }
        }
    }

    /// The bit-sliced test agrees with the brute force on random wing states and on states
    /// two moves before pairable ones.
    #[test]
    fn matches_brute_force() {
        let t = P1Tables::with_table(Staging::new(), Mod3Table::from_depths(&[0]));
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
        let quarters: Vec<usize> = (0..N_MOVES)
            .filter(|&m| (WIDE_QUARTER | OUTER_QUARTER) >> m & 1 == 1)
            .collect();
        let mut pass = [[0usize; 2]; 2];
        for i in 0..40000 {
            let (pos, odd) = if i % 2 == 0 {
                let mut p: Vec<u8> = (0..24).collect();
                p.shuffle(&mut rng);
                (V32::from_slice(&p), Cube::parity(&p) == 1)
            } else {
                // Two moves (the second a quarter turn) undone from the solved wings.
                let mut q = V32::from_slice(&std::array::from_fn::<u8, 24, _>(|k| k as u8));
                let ms = [
                    quarters[rand::Rng::gen_range(&mut rng, 0..quarters.len())],
                    rand::Rng::gen_range(&mut rng, 0..N_MOVES),
                ];
                for &m in &ms {
                    q = t.wing_dst[crate::moves::inverse_move(m)].gather(&q);
                }
                (
                    q,
                    (WIDE_QUARTER >> ms[0] & 1) ^ (WIDE_QUARTER >> ms[1] & 1) == 1,
                )
            };
            for tight in [true, false] {
                let want = possible_ref(&t, &pos, odd, tight);
                assert_eq!(t.wing2.possible(&pos, odd, tight), want, "case {i}");
                pass[i % 2][tight as usize] += want as usize;
            }
        }
        // Random states rarely pass; most of the constructed ones do.
        assert!(pass[0][1] < 1500 && pass[0][0] < 2500, "{pass:?}");
        assert!(pass[1][0] > 15000, "{pass:?}");
    }
}
