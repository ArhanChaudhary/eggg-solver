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
    /// The node-parallel form (see [`Wing2::possible_soa`]), per wing parity.
    soa: [SoaPlanes; 2],
}

/// The preimages of one wing parity in bit planes of 8 (the tight ones first, all frames),
/// for [`Wing2::possible_soa`].
pub struct SoaPlanes {
    /// Per plane, per slot: the plane's preimages (bits) that contain the slot.
    t: Vec<V32>,
    /// Per plane, the complement of its preimages' frames' `high_odd` bits (splat).
    nho: Vec<V32>,
    /// Per plane, its tight preimages (bits).
    tight: Vec<u8>,
    /// The planes that hold tight preimages (the first ones).
    n_tight: usize,
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
        let soa = [false, true].map(|odd| {
            // (preimage, frame, tight), the tight preimages (a subset of the slack ones) first.
            let mut all = vec![];
            for (f, fr) in frames.iter().enumerate() {
                let tight = Self::preimages(fr, wing_dst, true, odd);
                let slack = Self::preimages(fr, wing_dst, false, odd);
                assert!(tight.iter().all(|p| slack.contains(p)));
                all.extend(tight.iter().map(|&p| (p, f, true)));
                all.extend(
                    slack
                        .iter()
                        .filter(|p| !tight.contains(p))
                        .map(|&p| (p, f, false)),
                );
            }
            all.sort_by_key(|e| !e.2);
            // The slack-only preimages start a new plane (empty bits never pass).
            let n_tight = all.iter().filter(|e| e.2).count().div_ceil(8);
            while all.len() < 8 * n_tight || !all[..8 * n_tight].iter().all(|e| e.2) {
                all.insert(all.iter().filter(|e| e.2).count(), (0, 0, true));
            }
            let mut planes = SoaPlanes {
                t: vec![],
                nho: vec![],
                tight: vec![],
                n_tight,
            };
            for chunk in all.chunks(8) {
                let mut t = V32::ZERO;
                let mut ho = 0u8;
                let mut ti = 0u8;
                for (j, &(p, f, tight)) in chunk.iter().enumerate() {
                    for x in 0..24 {
                        if p >> x & 1 == 1 {
                            t.0[x] |= 1 << j;
                        }
                    }
                    if high_odd.0[f] != 0 {
                        ho |= 1 << j;
                    }
                    if tight {
                        ti |= 1 << j;
                    }
                }
                planes.t.push(t);
                planes.nho.push(V32([!ho; 32]));
                planes.tight.push(ti);
            }
            planes
        });
        Wing2 { b, high_odd, soa }
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

    /// [`Wing2::possible`] without bounds checks on the slots (each < 24).
    #[inline(always)]
    pub fn possible_unchecked(&self, pos: &V32, odd: bool, tight: bool) -> bool {
        let b = &self.b[!tight as usize][odd as usize];
        let mut ok = [!0u64; 3];
        let mut h = [0u64; 3];
        for e in 0..12 {
            // SAFETY: wing slots are < 24.
            let (x, y) = unsafe {
                debug_assert!(pos.0[2 * e] < 24 && pos.0[2 * e + 1] < 24);
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

    /// [`Wing2::possible`] for up to 32 nodes of wing parity `odd` at once: `p[w]` holds the
    /// slot of wing w of node i in byte i (the nodes' `pos` transposed, see
    /// [`crate::simd::transpose32`]); bit i of `tight` (of `lanes`, the nodes present) says
    /// whether node i is tight. Returns the nodes that pass (bits). The preimages are in bit
    /// planes of 8: per plane, a node's 24 wings are looked up in the plane's slot table (one
    /// byte gather per wing for all nodes), and the dedge-split and high-parity tests run on
    /// the 8 preimages of each byte at once.
    #[inline(always)]
    pub fn possible_soa(&self, p: &[V32], odd: bool, tight: u32, lanes: u32) -> u32 {
        use crate::simd::{and, gather, or, xor};
        let pl = &self.soa[odd as usize];
        let all_tight = tight & lanes == lanes;
        let n = if all_tight { pl.n_tight } else { pl.t.len() };
        // 0xff at the tight nodes.
        let tl = V32(std::array::from_fn(|i| {
            0u8.wrapping_sub((tight >> i & 1) as u8)
        }));
        let ones = V32([0xff; 32]);
        let mut any = V32::ZERO;
        for k in 0..n {
            let t = &pl.t[k];
            let mut ok = ones;
            let mut h = V32::ZERO;
            for e in 0..12 {
                let x = gather(t, &p[2 * e]);
                let y = gather(t, &p[2 * e + 1]);
                ok = and(&ok, &xor(&x, &y));
                h = xor(&h, &x);
            }
            let mut r = and(&ok, &xor(&h, &pl.nho[k]));
            if !all_tight {
                // A tight node counts only the plane's tight preimages.
                let ti = V32([pl.tight[k]; 32]);
                let m = or(&and(&tl, &ti), &xor(&tl, &ones));
                r = and(&r, &m);
            }
            any = or(&any, &r);
        }
        // Bit 7 set exactly at the nonzero bytes.
        let nz = crate::simd::add(&crate::simd::min(&any, &V32([1; 32])), &V32([0x7f; 32]));
        crate::simd::movemask(&nz) & lanes
    }

    /// Can a node with wing positions `pos` (the slot of each wing) and wing parity `odd`, two
    /// moves from the end of phase 1, reach an endpoint as far as its wings tell? `tight`: the
    /// node is at staging distance 2 (its last move is then a wide quarter turn).
    #[inline(always)]
    pub fn possible(&self, pos: &V32, odd: bool, tight: bool) -> bool {
        let b = &self.b[!tight as usize][odd as usize];
        // Per preimage (bit): does it split every dedge (ok), and the parity h of the number
        // of dedges whose first wing (2e) is in it. Where every dedge is split, a dedge's high
        // piece is in the preimage exactly when its first wing is, flipped if the high piece
        // is the second wing; so the high pieces in it number h plus the frame's count of
        // second-wing high pieces, mod 2, which must be even.
        let mut ok = [!0u64; 4];
        let mut h = [0u64; 4];
        for e in 0..12 {
            let x = &b[pos.0[2 * e] as usize].0;
            let y = &b[pos.0[2 * e + 1] as usize].0;
            for w in 0..4 {
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
    fn soa_matches_scalar() {
        let t = P1Tables::with_table(Staging::new(), Mod3Table::from_depths(&[0]));
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(11);
        let quarters: Vec<usize> = (0..N_MOVES)
            .filter(|&m| (WIDE_QUARTER | OUTER_QUARTER) >> m & 1 == 1)
            .collect();
        let mut passes = 0;
        for round in 0..3000 {
            let odd = round % 2 == 1;
            let n = 1 + round % 32;
            let mut m = [V32::ZERO; 32];
            let mut tight = 0u32;
            for (i, row) in m.iter_mut().enumerate().take(n) {
                let mut q = V32::from_slice(&std::array::from_fn::<u8, 24, _>(|k| k as u8));
                if i % 3 == 0 {
                    let mut p: Vec<u8> = (0..24).collect();
                    p.shuffle(&mut rng);
                    q = V32::from_slice(&p);
                } else {
                    for _ in 0..2 {
                        let mv = quarters[rand::Rng::gen_range(&mut rng, 0..quarters.len())];
                        q = t.wing_dst[crate::moves::inverse_move(mv)].gather(&q);
                    }
                    for _ in 0..(i % 4) {
                        let mv = rand::Rng::gen_range(&mut rng, 0..N_MOVES);
                        q = t.wing_dst[mv].gather(&q);
                    }
                }
                *row = q;
                if rand::Rng::gen_range(&mut rng, 0..3) != 0 {
                    tight |= 1 << i;
                }
            }
            let rows = m;
            crate::simd::transpose32(&mut m);
            let lanes = if n == 32 { !0 } else { (1u32 << n) - 1 };
            let got = t.wing2.possible_soa(&m[..24], odd, tight, lanes);
            for i in 0..n {
                let want = t.wing2.possible(&rows[i], odd, tight >> i & 1 == 1);
                assert_eq!(got >> i & 1 == 1, want, "round {round} node {i}");
                assert_eq!(
                    t.wing2
                        .possible_unchecked(&rows[i], odd, tight >> i & 1 == 1),
                    want
                );
                passes += want as usize;
            }
        }
        assert!(passes > 1000, "{passes}");
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

#[cfg(feature = "p1prof")]
thread_local! {
    static CAP: std::cell::RefCell<(usize, Vec<(V32, bool, bool)>)> =
        std::cell::RefCell::new((
            std::env::var("EGGG_W2_CAPTURE").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            vec![],
        ));
}

/// Profiling (feature `p1prof`): record the inputs of the lookahead tests, up to
/// `EGGG_W2_CAPTURE` of them, per thread (see [`take_capture`]).
#[cfg(feature = "p1prof")]
pub fn capture(pos: &V32, odd: bool, tight: bool) {
    CAP.with(|c| {
        let mut c = c.borrow_mut();
        if c.1.len() < c.0 {
            c.1.push((*pos, odd, tight));
        }
    });
}

/// The inputs recorded by [`capture`] on this thread.
#[cfg(feature = "p1prof")]
pub fn take_capture() -> Vec<(V32, bool, bool)> {
    CAP.with(|c| std::mem::take(&mut c.borrow_mut().1))
}
