//! The 3x3x3 reached after reduction: cubie model, moves (U R F D L B, indices 0..18 as in
//! the 4x4x4 move numbering), and extraction from a reduced 4x4x4.

use crate::cube::{CORNER_POS, Cube, EDGE_POS, GEO, Perm, V3, mat_apply, rot, rotation_mapping};
use crate::moves::{Conj, MOVE_PERMS, turn_of};
use std::sync::LazyLock;

pub const N_MOVES3: usize = 18;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cube3 {
    pub cp: [u8; 8],
    pub co: [u8; 8],
    pub ep: [u8; 12],
    pub eo: [u8; 12],
}

/// Reference sticker normal of each edge slot: U/D for the U and D layers, F/B for the E slice.
fn edge_ref(e: usize) -> V3 {
    let p = EDGE_POS[e];
    if p[1] != 0 {
        [0, p[1].signum(), 0]
    } else {
        [0, 0, p[2].signum()]
    }
}

/// Move tables: `new[t] = old[src[t]]`, with twist / flip added.
pub struct Move3 {
    pub corner_src: [u8; 8],
    pub corner_twist: [u8; 8],
    pub edge_src: [u8; 12],
    pub edge_flip: [u8; 12],
}

pub static MOVES3: LazyLock<Vec<Move3>> = LazyLock::new(|| {
    (0..N_MOVES3)
        .map(|m| {
            let p: &Perm = &MOVE_PERMS[m];
            let t = turn_of(m);
            let mut edge_src = [0u8; 12];
            let mut edge_flip = [0u8; 12];
            for s in 0..12 {
                let d = if t.moves(EDGE_POS[s]) {
                    rot(EDGE_POS[s], t.axis as usize, t.k)
                } else {
                    EDGE_POS[s]
                };
                let d = EDGE_POS.iter().position(|&q| q == d).unwrap();
                let r = if t.moves(EDGE_POS[s]) {
                    rot(edge_ref(s), t.axis as usize, t.k)
                } else {
                    edge_ref(s)
                };
                edge_src[d] = s as u8;
                edge_flip[d] = (r != edge_ref(d)) as u8;
            }
            Move3 {
                corner_src: p.corner_src,
                corner_twist: p.corner_twist,
                edge_src,
                edge_flip,
            }
        })
        .collect()
});

/// Orientation (0 or 1) of the dedge whose wing `p` sits in wing slot `s`.
static WING_EO: LazyLock<[[u8; 24]; 24]> = LazyLock::new(|| {
    let g = &*GEO;
    let mut t = [[0u8; 24]; 24];
    for s in 0..24 {
        for p in 0..24 {
            let r = rotation_mapping(g.wing_pos[p], g.wing_pos[s]);
            t[s][p] = (mat_apply(&r, edge_ref(p / 2)) != edge_ref(s / 2)) as u8;
        }
    }
    t
});

impl Cube3 {
    pub const SOLVED: Cube3 = Cube3 {
        cp: [0, 1, 2, 3, 4, 5, 6, 7],
        co: [0; 8],
        ep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        eo: [0; 12],
    };

    #[inline]
    pub fn apply(&self, m: usize) -> Cube3 {
        let mv = &MOVES3[m];
        let mut c = *self;
        for t in 0..8 {
            let s = mv.corner_src[t] as usize;
            c.cp[t] = self.cp[s];
            c.co[t] = (self.co[s] + mv.corner_twist[t]) % 3;
        }
        for t in 0..12 {
            let s = mv.edge_src[t] as usize;
            c.ep[t] = self.ep[s];
            c.eo[t] = self.eo[s] ^ mv.edge_flip[t];
        }
        c
    }

    pub fn apply_moves(&self, ms: &[u8]) -> Cube3 {
        ms.iter().fold(*self, |c, &m| c.apply(m as usize))
    }

    pub fn inverse(&self) -> Cube3 {
        let mut c = *self;
        for i in 0..8 {
            let p = self.cp[i] as usize;
            c.cp[p] = i as u8;
            c.co[p] = (3 - self.co[i]) % 3;
        }
        for i in 0..12 {
            let p = self.ep[i] as usize;
            c.ep[p] = i as u8;
            c.eo[p] = self.eo[i];
        }
        c
    }

    pub fn is_valid(&self) -> bool {
        Cube::parity(&self.cp) == Cube::parity(&self.ep)
            && self.co.iter().map(|&x| x as u32).sum::<u32>() % 3 == 0
            && self.eo.iter().map(|&x| x as u32).sum::<u32>() % 2 == 0
    }

    /// Extract the 3x3x3 from a reduced 4x4x4 (centres solved in some orientation, dedges
    /// paired). The result is relative to the centres, so solving it solves the 4x4x4.
    pub fn from_reduced(c: &Cube) -> Option<Cube3> {
        // Recolour so that the centres are in their home orientation.
        let cols = c.centre_colours();
        let face_colour: Vec<u8> = (0..6).map(|f| cols[f * 4]).collect();
        let mut x = *c;
        if face_colour != [0, 1, 2, 3, 4, 5] {
            let cj = crate::cube::ROTATIONS
                .iter()
                .map(|m| Conj::new(*m))
                .find(|cj| {
                    let map = cj.colour_map();
                    (0..6).all(|f| map[face_colour[f] as usize] as usize == f)
                })?;
            x = cj.recolour(c);
        }
        let cols = x.centre_colours();
        if (0..24).any(|s| cols[s] as usize != s / 4) {
            return None;
        }
        let mut out = Cube3 {
            cp: x.cp,
            co: x.co,
            ep: [0; 12],
            eo: [0; 12],
        };
        for e in 0..12 {
            let (a, b) = (x.wp[2 * e] as usize, x.wp[2 * e + 1] as usize);
            if a / 2 != b / 2 {
                return None;
            }
            out.ep[e] = (a / 2) as u8;
            out.eo[e] = WING_EO[2 * e][a];
            if WING_EO[2 * e + 1][b] != out.eo[e] {
                return None;
            }
        }
        out.is_valid().then_some(out)
    }
}

#[allow(unused)]
const _C: [V3; 8] = CORNER_POS;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn outer_moves_keep_reduction() {
        let mut rng = rand::thread_rng();
        let mut c = Cube::SOLVED;
        let mut c3 = Cube3::SOLVED;
        for _ in 0..100 {
            let m = rng.gen_range(0..18);
            c = c.apply_perm(&MOVE_PERMS[m]);
            c3 = c3.apply(m);
            assert_eq!(Cube3::from_reduced(&c), Some(c3));
            assert_eq!(c3.inverse().inverse(), c3);
        }
        // F flips four edges, U none.
        assert_eq!(Cube3::SOLVED.apply(6).eo.iter().sum::<u8>(), 4);
        assert_eq!(Cube3::SOLVED.apply(0).eo.iter().sum::<u8>(), 0);
    }
}
