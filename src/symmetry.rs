//! Symmetries of the position being solved, and the phase-1 jobs they make redundant.
//!
//! A spatial symmetry `s` (one of the 48: 24 rotations, each also composed with the central
//! inversion) acts on a facelet colouring `c` by moving every facelet: `(s.c)(s x) = c(x)`.
//! A position is symmetric under `s` when `s.c` is `c` up to a relabelling of the colours
//! (identical centres need no special care at this level). Moves are conjugated with it:
//! `(s.c).(s m s^-1) = s.(c.m)`, and a relabelling keeps every face single-coloured, so a
//! sequence `M` solves `c` exactly when `s M s^-1` does: the solutions of a symmetric position
//! come in orbits of equally long sequences.
//!
//! The solver's 27 moves (U R F D L B Uw Rw Fw) are mapped onto themselves by the six
//! symmetries that fix the URF corner (the turns by 120 and 240 degrees about the URF-DBL
//! diagonal and the three reflections that swap two of U, R and F, which also invert the turn
//! direction); the other 42 take some of Uw, Rw and Fw to Dw, Lw or Bw. These six also keep the
//! canonical move order (`ALLOWED_AFTER`: they map faces to faces and axes to axes, and keep
//! the order U < D < Uw within each axis), so the image of a canonical sequence is canonical,
//! and the image of its prefix is the prefix of its image. For the subgroup `S` of the
//! position's symmetries that fix URF, phase 1 searches only the job prefixes that are the
//! least (lexicographically, by move index) of their orbit under `S`: every solution `M` has
//! an image `s M s^-1` (`s` in `S`) of the same length whose prefix is that least one, and the
//! search below the image's prefix finds the images of the endpoints below the original one
//! (in the frame and colour assignment `s` takes theirs to; all are searched), so no solution
//! length is lost. The phase-2 and phase-3 work of the endpoints below the pruned prefixes goes
//! with them.
//!
//! Before that, the input is rotated (a whole-cube rotation, which the solver's notion of a
//! position ignores) so that the corner direction fixed by the most of its symmetries becomes
//! URF: a position symmetric about another diagonal is then pruned as well. The solutions are
//! mapped back, with any Dw, Lw or Bw that the back-rotation makes written as Uw, Rw or Fw
//! (the cube may end in any orientation), so they have the same length.

use crate::cube::{
    B, Cube, D, FACE_NORMAL, Facelets, GEO, IDENTITY, L, Mat, ROTATIONS, face_of_normal, mat_apply,
    mat_mul, rot,
};
use crate::moves::{MOVE_FACES, N_MOVES};
use std::sync::LazyLock;

/// The 48 symmetries of the cube: `ROTATIONS` (index 0 the identity), then each of them
/// composed with the central inversion (the improper ones: reflections and rotoreflections).
pub static SYMS: LazyLock<Vec<Mat>> = LazyLock::new(|| {
    let mut v = ROTATIONS.clone();
    v.extend(ROTATIONS.iter().map(|m| m.map(|row| row.map(|x| -x))));
    v
});

/// Per symmetry, the facelet each facelet goes to.
static FACELET_PERM: LazyLock<Vec<[u8; 96]>> = LazyLock::new(|| {
    let g = &*GEO;
    SYMS.iter()
        .map(|m| {
            std::array::from_fn(|i| {
                let (p, n) = g.facelet_pos[i];
                let to = (mat_apply(m, p), mat_apply(m, n));
                g.facelet_pos.iter().position(|&q| q == to).unwrap() as u8
            })
        })
        .collect()
});

const URF: [i8; 3] = [1, 1, 1];

pub fn transpose(m: &Mat) -> Mat {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}

fn det(m: &Mat) -> i32 {
    let m = m.map(|r| r.map(i32::from));
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The colouring moved by symmetry `s` (no relabelling): `(s.c)(s x) = c(x)`.
pub fn spatial(f: &Facelets, s: usize) -> Facelets {
    let mut out = [0u8; 96];
    for (i, &j) in FACELET_PERM[s].iter().enumerate() {
        out[j as usize] = f.0[i];
    }
    Facelets(out)
}

/// Is `f` invariant under symmetry `s` up to a relabelling of the colours?
pub fn is_symmetric(f: &Facelets, s: usize) -> bool {
    // (s.c)(j) = pi(c(j)) for all j: with j = perm(i), c(i) = pi(c(perm(i))).
    let mut pi = [u8::MAX; 6];
    for (i, &j) in FACELET_PERM[s].iter().enumerate() {
        let (from, to) = (f.0[j as usize] as usize, f.0[i]);
        if pi[from] == u8::MAX {
            pi[from] = to;
        } else if pi[from] != to {
            return false;
        }
    }
    let mut seen = 0u8;
    for &p in &pi {
        if p != u8::MAX {
            if seen >> p & 1 == 1 {
                return false;
            }
            seen |= 1 << p;
        }
    }
    true
}

/// The symmetries (indices into [`SYMS`]) under which `f` is invariant up to a relabelling of
/// the colours: its stabiliser.
pub fn stabiliser(f: &Facelets) -> Vec<usize> {
    (0..SYMS.len()).filter(|&s| is_symmetric(f, s)).collect()
}

/// The image of move `m` under a symmetry that fixes the URF corner (it maps the 27 moves onto
/// themselves; an improper one inverts the turn direction).
pub fn sym_move(m: usize, mat: &Mat) -> usize {
    debug_assert_eq!(mat_apply(mat, URF), URF);
    let (face, width) = MOVE_FACES[m / 3];
    let nf = face_of_normal(mat_apply(mat, FACE_NORMAL[face]));
    let i = MOVE_FACES.iter().position(|&x| x == (nf, width)).unwrap();
    let p = if det(mat) < 0 { 2 - m % 3 } else { m % 3 };
    i * 3 + p
}

/// The symmetry of a position as the solver uses it (see the module documentation).
#[derive(Clone, Debug)]
pub struct PositionSymmetry {
    /// The order of the position's stabiliser (1 to 48) and how many of its elements are
    /// rotations.
    pub order: usize,
    pub rotations: usize,
    /// The whole-cube rotation applied to the input before the search (`IDENTITY`: none).
    pub rotation: Mat,
    /// The move maps of the non-identity symmetries used to prune, in the rotated frame: the
    /// stabiliser's elements that fix the URF corner there.
    pub maps: Vec<[u8; N_MOVES]>,
}

impl PositionSymmetry {
    /// No symmetry used (and none detected).
    pub fn none() -> Self {
        PositionSymmetry {
            order: 1,
            rotations: 1,
            rotation: IDENTITY,
            maps: vec![],
        }
    }

    /// The stabiliser of `cube`; with `prune`, the rotation that makes the most of it fix URF
    /// and the move maps of those symmetries (else they are left empty).
    pub fn of(cube: &Cube, prune: bool) -> Self {
        let f = cube.to_facelets();
        let stab = stabiliser(&f);
        let mut out = PositionSymmetry {
            order: stab.len(),
            rotations: stab.iter().filter(|&&s| s < ROTATIONS.len()).count(),
            ..Self::none()
        };
        if !prune || stab.len() == 1 {
            return out;
        }
        // The corner direction fixed by the most symmetries (URF first, so that an input
        // with as many there keeps its orientation).
        let dirs: Vec<[i8; 3]> = (0..8)
            .map(|k| {
                let b = |i: usize| if (k >> i) & 1 == 0 { 1 } else { -1 };
                [b(0), b(1), b(2)]
            })
            .collect();
        let fixing = |v: [i8; 3]| -> Vec<usize> {
            stab.iter()
                .copied()
                .filter(|&s| mat_apply(&SYMS[s], v) == v)
                .collect()
        };
        let best = dirs.iter().copied().fold((URF, fixing(URF)), |acc, v| {
            let fx = fixing(v);
            if fx.len() > acc.1.len() { (v, fx) } else { acc }
        });
        let (v, fixed) = best;
        if fixed.len() == 1 {
            return out;
        }
        let rho = *ROTATIONS.iter().find(|m| mat_apply(m, v) == URF).unwrap();
        let rho_inv = transpose(&rho);
        out.rotation = rho;
        for &s in &fixed {
            if s == 0 {
                continue;
            }
            // rho s rho^-1 is a symmetry of the rotated position, fixing URF.
            let m = mat_mul(&mat_mul(&rho, &SYMS[s]), &rho_inv);
            out.maps
                .push(std::array::from_fn(|k| sym_move(k, &m) as u8));
        }
        out
    }

    /// The number of symmetries pruned by (1: none).
    pub fn used(&self) -> usize {
        self.maps.len() + 1
    }

    /// The input as the search sees it: `cube` turned by [`Self::rotation`] (its colours then
    /// relabelled so that the DBL corner is in place).
    pub fn rotated(&self, cube: &Cube) -> Cube {
        if self.rotation == IDENTITY {
            return *cube;
        }
        let r = ROTATIONS.iter().position(|m| *m == self.rotation).unwrap();
        spatial(&cube.to_facelets(), r).to_cube().unwrap().0
    }

    /// Is the job prefix `p` the least of its orbit under the symmetries used?
    pub fn keeps(&self, p: &[u8]) -> bool {
        self.maps.iter().all(|map| {
            p.iter()
                .map(|&m| map[m as usize])
                .cmp(p.iter().copied())
                .is_ge()
        })
    }

    /// The images of a solution of the rotated position under the symmetries used (also its
    /// solutions, of the same length).
    pub fn images(&self, moves: &[u8]) -> Vec<Vec<u8>> {
        self.maps
            .iter()
            .map(|map| moves.iter().map(|&m| map[m as usize]).collect())
            .collect()
    }

    /// A solution of the rotated position as a solution of the input, move for move: each
    /// move turned back by the rotation, a Dw, Lw or Bw written as Uw, Rw or Fw (the same turn
    /// followed by a whole-cube turn, which the later moves are turned back by too).
    pub fn map_back(&self, moves: &[u8]) -> Vec<u8> {
        if self.rotation == IDENTITY {
            return moves.to_vec();
        }
        let mut g = transpose(&self.rotation);
        moves
            .iter()
            .map(|&m| {
                let (face, width) = MOVE_FACES[m as usize / 3];
                let power = m % 3 + 1;
                let f = face_of_normal(mat_apply(&g, FACE_NORMAL[face]));
                let f = if width == 2 && [D, L, B].contains(&f) {
                    // Dw^p = Uw^p then the whole cube turned like D^p (`Turn::face`: D
                    // turns by +90 degrees about the y axis per quarter turn).
                    let axis = crate::cube::FACE_AXIS[f];
                    let k = (if FACE_NORMAL[f][axis] > 0 {
                        3 * power
                    } else {
                        power
                    }) % 4;
                    let turn: Mat = transpose(&std::array::from_fn(|j| {
                        let mut e = [0i8; 3];
                        e[j] = 1;
                        rot(e, axis, k)
                    }));
                    g = mat_mul(&transpose(&turn), &g);
                    (f + 3) % 6
                } else {
                    f
                };
                let i = MOVE_FACES.iter().position(|&x| x == (f, width)).unwrap();
                (i * 3) as u8 + power - 1
            })
            .collect()
    }

    /// A one-line description (verbose and debug output).
    pub fn describe(&self) -> String {
        let mut s = format!(
            "stabiliser of order {} ({} rotations, {} improper)",
            self.order,
            self.rotations,
            self.order - self.rotations
        );
        if self.used() > 1 {
            s += &format!(", pruning by {} symmetries fixing URF", self.used());
            if self.rotation != IDENTITY {
                s += " after a whole-cube rotation";
            }
        }
        s
    }
}

/// The index in [`SYMS`] of a symmetry matrix.
pub fn sym_index(m: &Mat) -> usize {
    SYMS.iter()
        .position(|x| x == m)
        .expect("not a cube symmetry")
}

/// The group generated by the symmetries `gens` (indices into [`SYMS`]), sorted.
pub fn group(gens: &[usize]) -> Vec<usize> {
    let mut g = vec![0usize];
    let mut i = 0;
    while i < g.len() {
        for &h in gens {
            let m = sym_index(&mat_mul(&SYMS[h], &SYMS[g[i]]));
            if !g.contains(&m) {
                g.push(m);
            }
        }
        i += 1;
    }
    g.sort();
    g
}

/// A random position invariant under the symmetry group `grp` (indices into [`SYMS`]; for
/// tests and benchmarks). Per orbit of piece slots (corners, wings, centres) one slot gets a
/// random unused piece in a random valid placement, and the orbit's other slots get its
/// conjugates: piece `s(p)` at slot `s(k)`, coloured `pi_s` of the colours at `k` (`pi_s` the
/// face map of `s`, under which the solved cube stays solved). Retried until the corner
/// twists add up.
pub fn random_symmetric(grp: &[usize], rng: &mut impl rand::Rng) -> Facelets {
    use crate::cube::{CORNER_POS, centre_slot, corner_slot, wing_slot};
    use rand::seq::SliceRandom;
    let g = &*GEO;
    let solved = Cube::SOLVED.to_facelets();
    let fidx = |p, n| g.facelet_pos.iter().position(|&q| q == (p, n)).unwrap();
    // Per kind (corners, wings, centres): slot positions and the facelets of each slot.
    let pos: [Vec<[i8; 3]>; 3] = [
        CORNER_POS.to_vec(),
        g.wing_pos.to_vec(),
        g.centre_pos.to_vec(),
    ];
    let facelets: [Vec<Vec<usize>>; 3] = [
        (0..8)
            .map(|k| {
                g.corner_normals[k]
                    .iter()
                    .map(|&n| fidx(CORNER_POS[k], n))
                    .collect()
            })
            .collect(),
        (0..24)
            .map(|k| {
                g.wing_normals[k]
                    .iter()
                    .map(|&n| fidx(g.wing_pos[k], n))
                    .collect()
            })
            .collect(),
        (0..24)
            .map(|k| {
                let p = g.centre_pos[k];
                let axis = p.iter().position(|c| c.abs() == 3).unwrap();
                let mut n = [0i8; 3];
                n[axis] = p[axis].signum();
                vec![fidx(p, n)]
            })
            .collect(),
    ];
    let slot_of = |kind: usize, p: [i8; 3]| match kind {
        0 => corner_slot(p),
        1 => wing_slot(p),
        _ => centre_slot(p),
    };
    let pi: Vec<[u8; 6]> = grp
        .iter()
        .map(|&s| {
            std::array::from_fn(|f| face_of_normal(mat_apply(&SYMS[s], FACE_NORMAL[f])) as u8)
        })
        .collect();
    // The solved cube turned by each rotation: its colours at slot k are a valid placement of
    // the piece whose home the rotation takes to k (every piece, every orientation).
    let turned: Vec<Facelets> = (0..24).map(|r| spatial(&solved, r)).collect();
    'retry: loop {
        let mut col = [u8::MAX; 96];
        let mut used: [Vec<bool>; 3] = [vec![false; 8], vec![false; 24], vec![false; 24]];
        let mut slots: Vec<(usize, usize)> = (0..3)
            .flat_map(|kind| (0..pos[kind].len()).map(move |k| (kind, k)))
            .collect();
        slots.shuffle(rng);
        for (kind, k) in slots {
            if col[facelets[kind][k][0]] != u8::MAX {
                continue;
            }
            let mut rs: Vec<usize> = (0..24).collect();
            rs.shuffle(rng);
            let mut placed = false;
            'cand: for r in rs {
                let home = slot_of(kind, mat_apply(&transpose(&ROTATIONS[r]), pos[kind][k]));
                let mut trial = col;
                let mut take = vec![];
                for (gi, &s) in grp.iter().enumerate() {
                    let piece = slot_of(kind, mat_apply(&SYMS[s], pos[kind][home]));
                    let to_slot = slot_of(kind, mat_apply(&SYMS[s], pos[kind][k]));
                    let fresh = trial[facelets[kind][to_slot][0]] == u8::MAX;
                    for &i in &facelets[kind][k] {
                        let to = FACELET_PERM[s][i] as usize;
                        let v = pi[gi][turned[r].0[i] as usize];
                        if trial[to] != u8::MAX && trial[to] != v {
                            continue 'cand;
                        }
                        trial[to] = v;
                    }
                    if fresh {
                        if used[kind][piece] || take.contains(&piece) {
                            continue 'cand;
                        }
                        take.push(piece);
                    }
                }
                for p in take {
                    used[kind][p] = true;
                }
                col = trial;
                placed = true;
                break;
            }
            if !placed {
                continue 'retry;
            }
        }
        let f = Facelets(col);
        if f.to_cube().is_ok() {
            return f;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::{ALLOWED_AFTER, turn_of};
    use rand::{Rng, SeedableRng};

    fn apply(f: &Facelets, moves: &[u8]) -> Facelets {
        moves
            .iter()
            .fold(*f, |f, &m| f.apply_turn(turn_of(m as usize)))
    }

    /// The six symmetries that fix URF.
    fn c3v() -> Vec<usize> {
        (0..48)
            .filter(|&s| mat_apply(&SYMS[s], URF) == URF)
            .collect()
    }

    #[test]
    fn symmetries_fixing_urf_map_the_moves_and_keep_canonical_order() {
        let grp = c3v();
        assert_eq!(grp.len(), 6);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(1);
        let f = Cube::random(&mut rng).to_facelets();
        for &s in &grp {
            let m = &SYMS[s];
            let map: Vec<usize> = (0..N_MOVES).map(|k| sym_move(k, m)).collect();
            let mut sorted = map.clone();
            sorted.sort();
            assert_eq!(sorted, (0..N_MOVES).collect::<Vec<_>>());
            for k in 0..N_MOVES {
                // (s.c).(s m s^-1) = s.(c.m)
                assert_eq!(
                    apply(&spatial(&f, s), &[map[k] as u8]),
                    spatial(&apply(&f, &[k as u8]), s),
                    "symmetry {s} move {k}"
                );
                for last in 0..=N_MOVES {
                    let ml = if last == N_MOVES { N_MOVES } else { map[last] };
                    assert_eq!(
                        ALLOWED_AFTER[last] >> k & 1,
                        ALLOWED_AFTER[ml] >> map[k] & 1
                    );
                }
            }
        }
        // The other 42 do not keep the move set.
        for s in 0..48 {
            let keeps = (0..N_MOVES).all(|k| {
                let (face, width) = MOVE_FACES[k / 3];
                let nf = face_of_normal(mat_apply(&SYMS[s], FACE_NORMAL[face]));
                MOVE_FACES.contains(&(nf, width))
            });
            assert_eq!(keeps, grp.contains(&s));
        }
    }

    #[test]
    fn stabilisers_of_symmetric_and_random_positions() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(2);
        for _ in 0..200 {
            let f = Cube::random(&mut rng).to_facelets();
            assert_eq!(stabiliser(&f), vec![0]);
            let sym = PositionSymmetry::of(&f.to_cube().unwrap().0, true);
            assert_eq!((sym.order, sym.used(), sym.rotation), (1, 1, IDENTITY));
        }
        let solved = Cube::SOLVED.to_facelets();
        assert_eq!(stabiliser(&solved).len(), 48);
        let grp = c3v();
        let three = grp[1..].iter().copied().find(|&s| s < 24).unwrap();
        let refl = grp.iter().copied().find(|&s| s >= 24).unwrap();
        // A turn by 90 degrees about U and the reflection R <-> L (not fixing URF).
        let quarter = sym_index(
            &ROTATIONS
                .iter()
                .copied()
                .find(|m| {
                    mat_apply(m, [0, 1, 0]) == [0, 1, 0] && mat_apply(m, [1, 0, 0]) == [0, 0, -1]
                })
                .unwrap(),
        );
        let mirror_x = sym_index(&[[-1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        // Another diagonal: the 3-fold turn about UFL-DBR (a rotation of `three`).
        let other3 = {
            let rho = ROTATIONS
                .iter()
                .copied()
                .find(|m| mat_apply(m, URF) == [-1, 1, 1])
                .unwrap();
            sym_index(&mat_mul(&mat_mul(&rho, &SYMS[three]), &transpose(&rho)))
        };
        for (gens, used) in [
            (vec![three], 3),
            (vec![refl], 2),
            (vec![three, refl], 6),
            (vec![other3], 3),
            (vec![quarter], 1),
            (vec![mirror_x], 1),
            (vec![quarter, mirror_x], 1),
        ] {
            let grp = group(&gens);
            for _ in 0..5 {
                let f = random_symmetric(&grp, &mut rng);
                let stab = stabiliser(&f);
                assert!(grp.iter().all(|s| stab.contains(s)), "{gens:?}: {stab:?}");
                let cube = f.to_cube().unwrap().0;
                let sym = PositionSymmetry::of(&cube, true);
                assert!(sym.used() >= used, "{gens:?}: {}", sym.describe());
                // The rotated position has the used symmetries.
                let rf = sym.rotated(&cube).to_facelets();
                for map in &sym.maps {
                    let s = (0..48)
                        .find(|&s| {
                            mat_apply(&SYMS[s], URF) == URF
                                && (0..N_MOVES).all(|k| sym_move(k, &SYMS[s]) == map[k] as usize)
                        })
                        .unwrap();
                    assert!(is_symmetric(&rf, s));
                }
            }
        }
        // A classic pattern: the checkerboard has all 48 symmetries.
        let turns = crate::notation::parse_moves("R2 L2 U2 D2 F2 B2").unwrap();
        let f = turns.iter().fold(solved, |f, &t| f.apply_turn(t));
        assert_eq!(stabiliser(&f).len(), 48);
        let sym = PositionSymmetry::of(&f.to_cube().unwrap().0, true);
        assert_eq!((sym.used(), sym.rotation), (6, IDENTITY));
    }

    /// The kept prefixes hold exactly one prefix of each orbit, and their orbits cover all.
    #[test]
    fn one_kept_prefix_per_orbit() {
        let grp = c3v();
        let three = grp[1..].iter().copied().find(|&s| s < 24).unwrap();
        let refl = grp.iter().copied().find(|&s| s >= 24).unwrap();
        for gens in [vec![three], vec![refl], vec![three, refl]] {
            let elems = group(&gens);
            let sym = PositionSymmetry {
                maps: elems[1..]
                    .iter()
                    .map(|&s| std::array::from_fn(|k| sym_move(k, &SYMS[s]) as u8))
                    .collect(),
                ..PositionSymmetry::none()
            };
            for depth in 1..=3 {
                let mut all: Vec<Vec<u8>> = vec![vec![]];
                for _ in 0..depth {
                    all = all
                        .iter()
                        .flat_map(|p| {
                            let last = p.last().map_or(N_MOVES, |&m| m as usize);
                            (0..N_MOVES as u8)
                                .filter(move |&m| ALLOWED_AFTER[last] >> m & 1 == 1)
                                .map(move |m| {
                                    let mut q = p.clone();
                                    q.push(m);
                                    q
                                })
                        })
                        .collect();
                }
                let set: std::collections::HashSet<Vec<u8>> = all.iter().cloned().collect();
                let mut covered = std::collections::HashSet::new();
                let mut kept = 0;
                for p in &all {
                    let orbit: std::collections::BTreeSet<Vec<u8>> =
                        std::iter::once(p.clone()).chain(sym.images(p)).collect();
                    assert!(orbit.iter().all(|q| set.contains(q)), "image not canonical");
                    let k = orbit.iter().filter(|q| sym.keeps(q)).count();
                    assert_eq!(k, 1, "{p:?}");
                    if sym.keeps(p) {
                        kept += 1;
                        covered.extend(orbit);
                    }
                }
                assert_eq!(covered.len(), all.len());
                let ratio = all.len() as f64 / kept as f64;
                assert!(
                    ratio > 0.8 * elems.len() as f64 || depth == 1,
                    "{gens:?} depth {depth}: {} of {}",
                    kept,
                    all.len()
                );
            }
        }
    }

    /// A solution of the rotated position, mapped back, solves the input (the two end states
    /// differ by a whole-cube rotation), move for move.
    #[test]
    fn mapped_back_sequences_match_the_rotated_ones() {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(3);
        for r in 0..24 {
            let cube = Cube::random(&mut rng);
            let f = cube.to_facelets();
            let sym = PositionSymmetry {
                rotation: ROTATIONS[r],
                ..PositionSymmetry::none()
            };
            let rf = spatial(&f, r);
            for _ in 0..10 {
                let n = rng.gen_range(0..30);
                let moves: Vec<u8> = (0..n).map(|_| rng.gen_range(0..N_MOVES as u8)).collect();
                let back = sym.map_back(&moves);
                assert_eq!(back.len(), moves.len());
                let a = apply(&rf, &moves);
                let b = apply(&f, &back);
                assert!(
                    (0..24).any(|t| spatial(&b, t) == a),
                    "rotation {r}: {moves:?} -> {back:?}"
                );
            }
            // And the rotated cube is the input turned by the rotation.
            let rc = sym.rotated(&cube).to_facelets();
            assert!(is_relabelling(&rc, &rf));
        }
    }

    /// Completeness at the phase-1 level: on symmetric positions (rotated as the solver does),
    /// the endpoints below the kept job prefixes and their images under the symmetries used
    /// are exactly all endpoints, frames included, and the kept prefixes need ~|S|x fewer
    /// nodes.
    #[test]
    fn kept_prefixes_find_every_endpoint_up_to_symmetry() {
        use crate::phase1::{P1State, Search1};
        use crate::solver::load_p1;
        use std::collections::BTreeSet;
        let t = load_p1(std::path::Path::new("tables"), false, 8, false);
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(4);
        let grp = c3v();
        let three = grp[1..].iter().copied().find(|&s| s < 24).unwrap();
        let refl = grp.iter().copied().find(|&s| s >= 24).unwrap();
        let other3 = {
            let rho = ROTATIONS
                .iter()
                .copied()
                .find(|m| mat_apply(m, URF) == [1, -1, 1])
                .unwrap();
            sym_index(&mat_mul(&mat_mul(&rho, &SYMS[three]), &transpose(&rho)))
        };
        let mut prefixes: Vec<Vec<u8>> = vec![vec![]];
        for _ in 0..3 {
            prefixes = prefixes
                .iter()
                .flat_map(|p| {
                    let last = p.last().map_or(N_MOVES, |&m| m as usize);
                    (0..N_MOVES as u8)
                        .filter(move |&m| ALLOWED_AFTER[last] >> m & 1 == 1)
                        .map(move |m| [p.as_slice(), &[m]].concat())
                })
                .collect();
        }
        for (gens, want) in [
            (vec![three], 3),
            (vec![refl], 2),
            (vec![three, refl], 6),
            (vec![other3], 3),
        ] {
            let f = random_symmetric(&group(&gens), &mut rng);
            let sym = PositionSymmetry::of(&f.to_cube().unwrap().0, true);
            assert_eq!(sym.used(), want);
            let cube = sym.rotated(&f.to_cube().unwrap().0);
            let root = P1State::from_cube(&cube);
            let d0 = t.distance(&root);
            // Frame f turns axis f (x, y, z) by half turns only: the axis map of each symmetry.
            let axes: Vec<[usize; 3]> = sym
                .maps
                .iter()
                .map(|map| {
                    let mut a = [0; 3];
                    for face in [crate::cube::U, crate::cube::R, crate::cube::F] {
                        let img = MOVE_FACES[map[face * 3] as usize / 3].0;
                        a[crate::cube::FACE_AXIS[face]] = crate::cube::FACE_AXIS[img];
                    }
                    a
                })
                .collect();
            let mut levels = 0;
            for len in d0.max(3)..d0 + 6 {
                let search = |keep: &dyn Fn(&[u8]) -> bool| {
                    let (mut eps, mut nodes) = (BTreeSet::new(), 0u64);
                    for p in prefixes.iter().filter(|p| keep(p)) {
                        let (st, d) = root.after(p, d0, &t);
                        if d > len - 3 {
                            continue;
                        }
                        let mut s = Search1::new(&t, |path: &[u8], fr| {
                            for g in 0..3 {
                                if fr >> g & 1 == 1 {
                                    eps.insert((path.to_vec(), g));
                                }
                            }
                            true
                        });
                        s.path.extend_from_slice(p);
                        s.dfs(&st, d, len - 3, p[2] as usize);
                        nodes += s.nodes;
                    }
                    (eps, nodes)
                };
                let (all, n_all) = search(&|_| true);
                if all.is_empty() {
                    continue;
                }
                let (kept, n_kept) = search(&|p| sym.keeps(p));
                let mut closure = kept.clone();
                for (path, g) in &kept {
                    for (map, ax) in sym.maps.iter().zip(&axes) {
                        closure.insert((
                            path.iter().map(|&m| map[m as usize]).collect(),
                            ax[*g as usize] as u8,
                        ));
                    }
                }
                assert_eq!(closure, all, "{gens:?} length {len}");
                let ratio = n_all as f64 / n_kept as f64;
                eprintln!(
                    "{gens:?} length {len}: endpoints {} kept {} nodes {n_all} / {n_kept} = {ratio:.2}",
                    all.len(),
                    kept.len()
                );
                assert!(ratio > 0.85 * want as f64, "{gens:?} length {len}: {ratio}");
                levels += 1;
                if levels == 2 || n_all > 20_000_000 {
                    break;
                }
            }
            assert!(levels > 0);
        }
    }

    fn is_relabelling(a: &Facelets, b: &Facelets) -> bool {
        let mut pi = [u8::MAX; 6];
        (0..96).all(|i| {
            let (x, y) = (a.0[i] as usize, b.0[i]);
            if pi[x] == u8::MAX {
                pi[x] = y;
            }
            pi[x] == y
        })
    }
}
