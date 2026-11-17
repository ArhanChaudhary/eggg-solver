//! The solver's move set and the symmetries it uses.
//!
//! The solver works in a fixed frame with 27 moves: U R F D L B Uw Rw Fw, each clockwise, half
//! and anticlockwise. Dw, Lw and Bw are never used: Dw is Uw followed by a rotation, and the
//! cube may end solved in any orientation, so leaving them out loses nothing and keeps the
//! frame fixed (the inner layer next to D, L and B never turns by itself).
//!
//! Move index `m = 3 * face + power - 1`, with faces in the order U R F D L B Uw Rw Fw.

use crate::cube::{
    B, Cube, D, F, FACE_NORMAL, L, Mat, Perm, R, ROTATIONS, Turn, U, face_of_normal, mat_apply,
};
use crate::notation::Move;
use std::sync::LazyLock;

pub const N_MOVES: usize = 27;

/// The nine move faces: (face, width).
pub const MOVE_FACES: [(usize, u8); 9] = [
    (U, 1),
    (R, 1),
    (F, 1),
    (D, 1),
    (L, 1),
    (B, 1),
    (U, 2),
    (R, 2),
    (F, 2),
];

pub fn move_of(m: usize) -> Move {
    let (face, width) = MOVE_FACES[m / 3];
    Move {
        face: face as u8,
        width,
        power: (m % 3) as u8 + 1,
    }
}

pub fn turn_of(m: usize) -> Turn {
    move_of(m).to_turn()
}

pub fn inverse_move(m: usize) -> usize {
    m / 3 * 3 + 2 - m % 3
}

/// Moves that may follow move `last` in a canonical sequence (`last = N_MOVES` at the start):
/// not the same face, and moves of one axis in increasing face order.
pub const ALLOWED_AFTER: [u32; N_MOVES + 1] = {
    let mut t = [0u32; N_MOVES + 1];
    let mut last = 0;
    while last <= N_MOVES {
        let mut m = 0;
        while m < N_MOVES {
            let ok = if last == N_MOVES {
                true
            } else {
                let (fl, fm) = (last / 3, m / 3);
                fl != fm && (fl % 3 != fm % 3 || fm > fl)
            };
            if ok {
                t[last] |= 1 << m;
            }
            m += 1;
        }
        last += 1;
    }
    t
};

/// Wide quarter turns: these (and only these) change the parity of the wing permutation.
pub const WIDE_QUARTER: u32 = (1 << 18) | (1 << 20) | (1 << 21) | (1 << 23) | (1 << 24) | (1 << 26);

/// Phase-2 moves in the standard frame (R/L axis special): U D F B, R2 L2, Uw2 Rw2 Fw2.
pub const PHASE2_MASK: u32 = 0b111
    | (0b111 << 9)
    | (0b111 << 6)
    | (0b111 << 15)
    | (1 << 4)
    | (1 << 13)
    | (1 << 19)
    | (1 << 22)
    | (1 << 25);

pub static MOVE_PERMS: LazyLock<[Perm; N_MOVES]> =
    LazyLock::new(|| std::array::from_fn(|m| Perm::of_turn(turn_of(m))));

/// The rotation that takes frame `f` to the standard frame (R/L axis special): f = 0 the
/// identity, f = 1 maps the U/D axis to R/L, f = 2 the F/B axis. These are the turns about the
/// UFR-DBL diagonal (by 0, 120 or 240 degrees), which map the 27-move set onto itself.
pub fn frame_rotation(f: usize) -> Mat {
    let src = [[1i8, 0, 0], [0, 1, 0], [0, 0, 1]][f];
    *ROTATIONS
        .iter()
        .find(|m| mat_apply(m, src) == [1, 0, 0] && mat_apply(m, [1, 1, 1]) == [1, 1, 1])
        .unwrap()
}

/// Image of move `m` under the spatial rotation `rot` (which must preserve the move set).
pub fn rotate_move(m: usize, rot: &Mat) -> usize {
    let (face, width) = MOVE_FACES[m / 3];
    let nf = face_of_normal(mat_apply(rot, FACE_NORMAL[face]));
    let i = MOVE_FACES
        .iter()
        .position(|&x| x == (nf, width))
        .expect("rotation does not preserve the move set");
    i * 3 + m % 3
}

/// A spatial symmetry applied to whole cube states: conjugation, `X -> s X s^-1`.
/// Moves on `X` correspond to the rotated moves on the conjugate.
pub struct Conj {
    corner_slot: [u8; 8],
    corner_twist: [u8; 8],
    wing_slot: [u8; 24],
    centre_slot: [u8; 24],
    colour: [u8; 6],
}

impl Conj {
    pub fn new(mat: Mat) -> Conj {
        use crate::cube::{CORNER_POS, GEO, centre_slot, corner_slot, wing_slot};
        let g = &*GEO;
        let mut c = Conj {
            corner_slot: [0; 8],
            corner_twist: [0; 8],
            wing_slot: [0; 24],
            centre_slot: [0; 24],
            colour: [0; 6],
        };
        for s in 0..8 {
            let d = corner_slot(mat_apply(&mat, CORNER_POS[s]));
            c.corner_slot[s] = d as u8;
            let n0 = mat_apply(&mat, g.corner_normals[s][0]);
            c.corner_twist[s] = g.corner_normals[d].iter().position(|&n| n == n0).unwrap() as u8;
        }
        for s in 0..24 {
            c.wing_slot[s] = wing_slot(mat_apply(&mat, g.wing_pos[s])) as u8;
            c.centre_slot[s] = centre_slot(mat_apply(&mat, g.centre_pos[s])) as u8;
        }
        for f in 0..6 {
            c.colour[f] = face_of_normal(mat_apply(&mat, FACE_NORMAL[f])) as u8;
        }
        c
    }

    /// The conjugate state: every piece and its slot are moved by the rotation.
    pub fn apply(&self, x: &Cube) -> Cube {
        let mut y = *x;
        for s in 0..8 {
            let d = self.corner_slot[s] as usize;
            let p = x.cp[s] as usize;
            y.cp[d] = self.corner_slot[p];
            // orientation: piece twist relative to slot reference stickers is preserved up to
            // the reference-sticker shift of the piece and of the slot.
            y.co[d] = (x.co[s] + self.corner_twist[s] + 3 - self.corner_twist[p]) % 3;
        }
        for s in 0..24 {
            y.wp[self.wing_slot[s] as usize] = self.wing_slot[x.wp[s] as usize];
            let c = x.ct[s];
            y.ct[self.centre_slot[s] as usize] = self.colour[(c / 4) as usize] * 4 + c % 4;
        }
        y
    }

    /// Recolour by this rotation without moving anything: every piece is renamed to the piece
    /// the rotation maps it to. Solutions of the recoloured cube solve the original.
    pub fn recolour(&self, x: &Cube) -> Cube {
        let mut y = *x;
        for s in 0..8 {
            let p = x.cp[s] as usize;
            y.cp[s] = self.corner_slot[p];
            y.co[s] = (x.co[s] + 3 - self.corner_twist[p]) % 3;
        }
        for s in 0..24 {
            y.wp[s] = self.wing_slot[x.wp[s] as usize];
            let c = x.ct[s];
            y.ct[s] = self.colour[(c / 4) as usize] * 4 + c % 4;
        }
        y
    }

    pub fn colour_map(&self) -> [u8; 6] {
        self.colour
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn canonical_rules() {
        // U then D allowed, D then U not; U then U not; U then Uw allowed.
        assert!(ALLOWED_AFTER[0] & (1 << 9) != 0);
        assert!(ALLOWED_AFTER[9] & 1 == 0);
        assert!(ALLOWED_AFTER[0] & 0b111 == 0);
        assert!(ALLOWED_AFTER[0] & (1 << 18) != 0);
        assert_eq!(PHASE2_MASK.count_ones(), 17);
    }

    #[test]
    fn frames_preserve_moves_and_conjugation_commutes() {
        let mut rng = rand::thread_rng();
        for f in 0..3 {
            let rot = frame_rotation(f);
            let conj = Conj::new(rot);
            let x = Cube::random(&mut rng);
            for m in 0..N_MOVES {
                let m2 = rotate_move(m, &rot);
                let a = conj.apply(&x.apply_perm(&MOVE_PERMS[m]));
                let b = conj.apply(&x).apply_perm(&MOVE_PERMS[m2]);
                assert_eq!(a, b, "frame {f} move {m}");
            }
        }
    }

    #[test]
    fn recolour_commutes_with_moves() {
        let mut rng = rand::thread_rng();
        for r in ROTATIONS.iter() {
            let c = Conj::new(*r);
            let x = Cube::random(&mut rng);
            let m = rng.gen_range(0..N_MOVES);
            assert_eq!(
                c.recolour(&x).apply_perm(&MOVE_PERMS[m]),
                c.recolour(&x.apply_perm(&MOVE_PERMS[m]))
            );
            // Recolouring keeps facelet solvedness: recolour of solved is solved up to rotation.
            assert!(c.recolour(&Cube::SOLVED).to_facelets().is_solved());
        }
    }
}
