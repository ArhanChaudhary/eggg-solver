//! Geometry of the 4x4x4 and the cubie-level model.
//!
//! Everything (piece slots, facelets, move permutations) is derived from 3D coordinates so that
//! there are no hand-written permutation tables to get wrong. Coordinates are doubled so that
//! they are integers: every cubie centre lies in {-3,-1,1,3}^3. Axes: x points to R, y to U,
//! z to F.

use std::sync::LazyLock;

pub type V3 = [i8; 3];

/// Faces, in facelet-string order.
pub const U: usize = 0;
pub const R: usize = 1;
pub const F: usize = 2;
pub const D: usize = 3;
pub const L: usize = 4;
pub const B: usize = 5;
pub const FACE_CHARS: [char; 6] = ['U', 'R', 'F', 'D', 'L', 'B'];

/// Outward normal of each face.
pub const FACE_NORMAL: [V3; 6] = [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
];

/// Axis (0 = x, 1 = y, 2 = z) of each face.
pub const FACE_AXIS: [usize; 6] = [1, 0, 2, 1, 0, 2];

pub fn face_of_normal(n: V3) -> usize {
    FACE_NORMAL
        .iter()
        .position(|&m| m == n)
        .expect("not a face normal")
}

/// Rotation by `k` times +90 degrees (right-hand rule) about axis `axis`.
pub fn rot(v: V3, axis: usize, k: u8) -> V3 {
    let mut v = v;
    for _ in 0..k % 4 {
        v = match axis {
            0 => [v[0], -v[2], v[1]],
            1 => [v[2], v[1], -v[0]],
            _ => [-v[1], v[0], v[2]],
        };
    }
    v
}

/// A 3x3 rotation matrix (rows), integer entries.
pub type Mat = [[i8; 3]; 3];

pub fn mat_apply(m: &Mat, v: V3) -> V3 {
    let mut r = [0i8; 3];
    for i in 0..3 {
        r[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
    }
    r
}

pub fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    let mut r = [[0i8; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

pub const IDENTITY: Mat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/// The matrix of the rotation by `k` quarter turns about the positive `axis`.
pub fn mat_of_rot(axis: usize, k: u8) -> Mat {
    let cols = [
        rot([1, 0, 0], axis, k),
        rot([0, 1, 0], axis, k),
        rot([0, 0, 1], axis, k),
    ];
    let mut m = [[0i8; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = cols[j][i];
        }
    }
    m
}

/// The 24 proper rotations of the cube. Index 0 is the identity.
pub static ROTATIONS: LazyLock<Vec<Mat>> = LazyLock::new(|| {
    let mut out = vec![IDENTITY];
    let mut i = 0;
    while i < out.len() {
        for axis in 0..3 {
            let m = mat_mul(&mat_of_rot(axis, 1), &out[i]);
            if !out.contains(&m) {
                out.push(m);
            }
        }
        i += 1;
    }
    assert_eq!(out.len(), 24);
    out
});

/// A layer turn: rotate every cubie whose coordinate along `axis` is in `layers` (bit i means
/// coordinate -3 + 2i) by `k` times +90 degrees about the positive axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Turn {
    pub axis: u8,
    pub layers: u8,
    pub k: u8,
}

impl Turn {
    /// Face turn notation: `face` turned clockwise `power` times (1..=3), `width` layers deep.
    pub fn face(face: usize, width: u8, power: u8) -> Turn {
        let axis = FACE_AXIS[face] as u8;
        let positive = FACE_NORMAL[face][axis as usize] > 0;
        let mut layers = 0u8;
        for w in 0..width {
            layers |= if positive { 8 >> w } else { 1 << w };
        }
        // Clockwise seen from outside is -90 degrees about the outward normal.
        let k = if positive { (3 * power) % 4 } else { power % 4 };
        Turn { axis, layers, k }
    }

    /// A single inner slice, `depth` layers in from `face` (depth 2 = the slice next to it).
    pub fn slice(face: usize, depth: u8, power: u8) -> Turn {
        let mut t = Turn::face(face, 1, power);
        let positive = FACE_NORMAL[face][t.axis as usize] > 0;
        t.layers = if positive {
            8 >> (depth - 1)
        } else {
            1 << (depth - 1)
        };
        t
    }

    /// Whole-cube rotation like `face` (x = R, y = U, z = F).
    pub fn rotation(face: usize, power: u8) -> Turn {
        let mut t = Turn::face(face, 1, power);
        t.layers = 15;
        t
    }

    pub fn inverse(self) -> Turn {
        Turn {
            k: (4 - self.k) % 4,
            ..self
        }
    }

    pub fn moves(&self, v: V3) -> bool {
        let c = v[self.axis as usize];
        let bit = ((c + 3) / 2) as u8;
        self.layers >> bit & 1 == 1
    }

    pub fn apply(&self, v: V3) -> V3 {
        if self.moves(v) {
            rot(v, self.axis as usize, self.k)
        } else {
            v
        }
    }
}

/// Corner slots in Kociemba order URF UFL ULB UBR DFR DLF DBL DRB.
pub const CORNER_POS: [V3; 8] = [
    [3, 3, 3],
    [-3, 3, 3],
    [-3, 3, -3],
    [3, 3, -3],
    [3, -3, 3],
    [-3, -3, 3],
    [-3, -3, -3],
    [3, -3, -3],
];
pub const DBL: usize = 6;

/// Edge slots in Kociemba order UR UF UL UB DR DF DL DB FR FL BL BR; each holds two wings.
pub const EDGE_POS: [V3; 12] = [
    [3, 3, 0],
    [0, 3, 3],
    [-3, 3, 0],
    [0, 3, -3],
    [3, -3, 0],
    [0, -3, 3],
    [-3, -3, 0],
    [0, -3, -3],
    [3, 0, 3],
    [-3, 0, 3],
    [-3, 0, -3],
    [3, 0, -3],
];

/// Geometric data shared by everything else, computed once.
pub struct Geometry {
    /// Corner slot stickers: normals ordered with the U/D sticker first, then clockwise.
    pub corner_normals: [[V3; 3]; 8],
    /// Wing slot positions: wing 2e+h lies on edge e, h = 0 for the end with the negative
    /// coordinate along the edge.
    pub wing_pos: [V3; 24],
    pub wing_normals: [[V3; 2]; 24],
    /// Centre slot positions: 4 per face in face order, reading order within the face.
    pub centre_pos: [V3; 24],
    /// Facelet positions and normals, 16 per face in face order, reading order.
    pub facelet_pos: [(V3, V3); 96],
}

fn facelet_geometry(face: usize, r: i8, c: i8) -> (V3, V3) {
    let pos = match face {
        U => [-3 + 2 * c, 3, -3 + 2 * r],
        R => [3, 3 - 2 * r, 3 - 2 * c],
        F => [-3 + 2 * c, 3 - 2 * r, 3],
        D => [-3 + 2 * c, -3, 3 - 2 * r],
        L => [-3, 3 - 2 * r, -3 + 2 * c],
        _ => [3 - 2 * c, 3 - 2 * r, -3],
    };
    (pos, FACE_NORMAL[face])
}

fn det(a: V3, b: V3, c: V3) -> i32 {
    let (a, b, c) = (a.map(i32::from), b.map(i32::from), c.map(i32::from));
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Outward normals of the stickers of the cubie at `p`.
fn normals(p: V3) -> Vec<V3> {
    let mut out = vec![];
    for (axis, &c) in p.iter().enumerate() {
        if c.abs() == 3 {
            let mut n = [0i8; 3];
            n[axis] = c.signum();
            out.push(n);
        }
    }
    out
}

pub static GEO: LazyLock<Geometry> = LazyLock::new(|| {
    let mut corner_normals = [[[0i8; 3]; 3]; 8];
    for (i, &p) in CORNER_POS.iter().enumerate() {
        let ns = normals(p);
        let ud = ns[1]; // y-normal (axes are in order x, y, z)
        let (a, b) = (ns[0], ns[2]);
        corner_normals[i] = if det(ud, a, b) == -1 {
            [ud, a, b]
        } else {
            [ud, b, a]
        };
    }
    let mut wing_pos = [[0i8; 3]; 24];
    let mut wing_normals = [[[0i8; 3]; 2]; 24];
    for (e, &p) in EDGE_POS.iter().enumerate() {
        let axis = p.iter().position(|&c| c == 0).unwrap();
        for h in 0..2 {
            let mut w = p;
            w[axis] = if h == 0 { -1 } else { 1 };
            wing_pos[2 * e + h] = w;
            let ns = normals(w);
            wing_normals[2 * e + h] = [ns[0], ns[1]];
        }
    }
    let mut facelet_pos = [([0i8; 3], [0i8; 3]); 96];
    let mut centre_pos = [[0i8; 3]; 24];
    for f in 0..6 {
        let mut ci = 0;
        for r in 0..4 {
            for c in 0..4 {
                let g = facelet_geometry(f, r, c);
                facelet_pos[f * 16 + (r * 4 + c) as usize] = g;
                if (1..3).contains(&r) && (1..3).contains(&c) {
                    centre_pos[f * 4 + ci] = g.0;
                    ci += 1;
                }
            }
        }
    }
    Geometry {
        corner_normals,
        wing_pos,
        wing_normals,
        centre_pos,
        facelet_pos,
    }
});

pub fn corner_slot(p: V3) -> usize {
    CORNER_POS.iter().position(|&q| q == p).unwrap()
}
pub fn wing_slot(p: V3) -> usize {
    GEO.wing_pos.iter().position(|&q| q == p).unwrap()
}
pub fn centre_slot(p: V3) -> usize {
    GEO.centre_pos.iter().position(|&q| q == p).unwrap()
}

/// Permutation form of a turn: `new[t] = old[src[t]]` (and corner twist added).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Perm {
    pub corner_src: [u8; 8],
    pub corner_twist: [u8; 8],
    pub wing_src: [u8; 24],
    pub centre_src: [u8; 24],
}

impl Perm {
    pub fn of_turn(t: Turn) -> Perm {
        let g = &*GEO;
        let mut p = Perm {
            corner_src: [0; 8],
            corner_twist: [0; 8],
            wing_src: [0; 24],
            centre_src: [0; 24],
        };
        for s in 0..8 {
            let d = corner_slot(t.apply(CORNER_POS[s]));
            let n0 = rot(
                g.corner_normals[s][0],
                t.axis as usize,
                if t.moves(CORNER_POS[s]) { t.k } else { 0 },
            );
            let j = g.corner_normals[d].iter().position(|&n| n == n0).unwrap();
            p.corner_src[d] = s as u8;
            p.corner_twist[d] = j as u8;
        }
        for s in 0..24 {
            p.wing_src[wing_slot(t.apply(g.wing_pos[s]))] = s as u8;
            p.centre_src[centre_slot(t.apply(g.centre_pos[s]))] = s as u8;
        }
        p
    }
}

/// Cubie-level state. Slots index the arrays; the values are the pieces sitting there.
/// Centre piece `i` has the colour of face `i / 4`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cube {
    pub cp: [u8; 8],
    pub co: [u8; 8],
    pub wp: [u8; 24],
    pub ct: [u8; 24],
}

impl Default for Cube {
    fn default() -> Self {
        Self::SOLVED
    }
}

const fn iota<const N: usize>() -> [u8; N] {
    let mut a = [0u8; N];
    let mut i = 0;
    while i < N {
        a[i] = i as u8;
        i += 1;
    }
    a
}

impl Cube {
    pub const SOLVED: Cube = Cube {
        cp: iota(),
        co: [0; 8],
        wp: iota(),
        ct: iota(),
    };

    pub fn apply_perm(&self, p: &Perm) -> Cube {
        let mut c = *self;
        for t in 0..8 {
            let s = p.corner_src[t] as usize;
            c.cp[t] = self.cp[s];
            c.co[t] = (self.co[s] + p.corner_twist[t]) % 3;
        }
        for t in 0..24 {
            c.wp[t] = self.wp[p.wing_src[t] as usize];
            c.ct[t] = self.ct[p.centre_src[t] as usize];
        }
        c
    }

    pub fn apply_turn(&self, t: Turn) -> Cube {
        self.apply_perm(&Perm::of_turn(t))
    }

    pub fn apply_turns(&self, ts: &[Turn]) -> Cube {
        ts.iter().fold(*self, |c, &t| c.apply_turn(t))
    }

    /// The inverse state: applying a move sequence to it gives the solved cube exactly when
    /// the inverse sequence turns the solved cube into this one.
    pub fn inverse(&self) -> Cube {
        let mut c = Cube::SOLVED;
        for s in 0..8 {
            let p = self.cp[s] as usize;
            c.cp[p] = s as u8;
            c.co[p] = (3 - self.co[s]) % 3;
        }
        for s in 0..24 {
            c.wp[self.wp[s] as usize] = s as u8;
            c.ct[self.ct[s] as usize] = s as u8;
        }
        c
    }

    /// Colour (face index) of the centre in each centre slot.
    pub fn centre_colours(&self) -> [u8; 24] {
        self.ct.map(|c| c / 4)
    }

    /// Solved up to the (indistinguishable) order of the centres of each colour.
    pub fn is_solved(&self) -> bool {
        self.cp == Cube::SOLVED.cp
            && self.co == [0; 8]
            && self.wp == Cube::SOLVED.wp
            && self.centre_colours() == Cube::SOLVED.centre_colours()
    }

    pub fn to_facelets(&self) -> Facelets {
        let g = &*GEO;
        let mut out = [0u8; 96];
        let fidx = |pos: V3, n: V3| g.facelet_pos.iter().position(|&q| q == (pos, n)).unwrap();
        for s in 0..8 {
            let p = self.cp[s] as usize;
            let o = self.co[s] as usize;
            for i in 0..3 {
                let colour = face_of_normal(g.corner_normals[p][i]);
                out[fidx(CORNER_POS[s], g.corner_normals[s][(i + o) % 3])] = colour as u8;
            }
        }
        for s in 0..24 {
            let p = self.wp[s] as usize;
            let rm = rotation_mapping(g.wing_pos[p], g.wing_pos[s]);
            for &n in &g.wing_normals[p] {
                out[fidx(g.wing_pos[s], mat_apply(&rm, n))] = face_of_normal(n) as u8;
            }
            let face = face_of_normal(normals(g.centre_pos[s])[0]);
            out[fidx(g.centre_pos[s], FACE_NORMAL[face])] = self.ct[s] / 4;
        }
        Facelets(out)
    }

    /// Parity (0 even, 1 odd) of a permutation.
    pub fn parity(p: &[u8]) -> u8 {
        let mut seen = [false; 32];
        let mut par = 0;
        for i in 0..p.len() {
            if !seen[i] {
                let mut j = i;
                let mut len = 0;
                while !seen[j] {
                    seen[j] = true;
                    j = p[j] as usize;
                    len += 1;
                }
                par ^= (len + 1) & 1;
            }
        }
        par as u8
    }

    /// Uniformly random state with the DBL corner solved (every position of the cube, up to
    /// rotation, has exactly one such representative).
    pub fn random<Rn: rand::Rng>(rng: &mut Rn) -> Cube {
        use rand::seq::SliceRandom;
        let mut c = Cube::SOLVED;
        let mut others: Vec<u8> = (0..8).filter(|&i| i != DBL as u8).collect();
        others.shuffle(rng);
        let mut sum = 0;
        let mut it = others.into_iter();
        for i in 0..8 {
            if i == DBL {
                continue;
            }
            c.cp[i] = it.next().unwrap();
            c.co[i] = rng.gen_range(0..3);
            sum += c.co[i];
        }
        let last = if DBL == 7 { 5 } else { 7 };
        c.co[last] = (3 + c.co[last] - sum % 3) % 3;
        c.wp.shuffle(rng);
        c.ct.shuffle(rng);
        c
    }
}

/// The rotation mapping position `a` to position `b` (unique for wings and corners up to the
/// stabiliser; the first found is returned).
pub fn rotation_mapping(a: V3, b: V3) -> Mat {
    *ROTATIONS
        .iter()
        .find(|m| mat_apply(m, a) == b)
        .expect("no rotation maps a to b")
}

/// Sticker colours (face indices 0..6), 16 per face in U R F D L B order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Facelets(pub [u8; 96]);

impl Facelets {
    pub fn parse(s: &str) -> Result<Facelets, String> {
        let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
        if chars.len() != 96 {
            return Err(format!("expected 96 facelets, got {}", chars.len()));
        }
        // Accept either face letters or any six symbols, mapped by the colour of each face's
        // first centre facelet.
        let mut out = [0u8; 96];
        let letters: String = FACE_CHARS.iter().collect();
        if chars
            .iter()
            .all(|c| letters.contains(c.to_ascii_uppercase()))
        {
            for (i, c) in chars.iter().enumerate() {
                out[i] = FACE_CHARS
                    .iter()
                    .position(|&f| f == c.to_ascii_uppercase())
                    .unwrap() as u8;
            }
        } else {
            let mut map: Vec<char> = vec![];
            for f in 0..6 {
                let c = chars[f * 16 + 5];
                if map.contains(&c) {
                    return Err("ambiguous colours: two faces share a centre colour".into());
                }
                map.push(c);
            }
            for (i, c) in chars.iter().enumerate() {
                out[i] = map
                    .iter()
                    .position(|m| m == c)
                    .ok_or(format!("unknown colour {c}"))? as u8;
            }
        }
        Ok(Facelets(out))
    }

    pub fn to_string_letters(&self) -> String {
        self.0.iter().map(|&c| FACE_CHARS[c as usize]).collect()
    }

    /// Every face a single colour.
    pub fn is_solved(&self) -> bool {
        (0..6).all(|f| {
            self.0[f * 16..f * 16 + 16]
                .iter()
                .all(|&c| c == self.0[f * 16])
        })
    }

    pub fn apply_turn(&self, t: Turn) -> Facelets {
        let g = &*GEO;
        let mut out = [0u8; 96];
        for i in 0..96 {
            let (p, n) = g.facelet_pos[i];
            let (q, m) = if t.moves(p) {
                (rot(p, t.axis as usize, t.k), rot(n, t.axis as usize, t.k))
            } else {
                (p, n)
            };
            let j = g.facelet_pos.iter().position(|&x| x == (q, m)).unwrap();
            out[j] = self.0[i];
        }
        Facelets(out)
    }

    /// Convert to the cubie model. The colours of the corner at DBL need not be D, B, L: the
    /// colours are first relabelled by the rotation that makes them so, and that relabelling
    /// (old colour -> new colour) is returned too.
    pub fn to_cube(&self) -> Result<(Cube, [u8; 6]), String> {
        let g = &*GEO;
        let at = |pos: V3, n: V3| {
            self.0[g.facelet_pos.iter().position(|&q| q == (pos, n)).unwrap()] as usize
        };
        for f in 0..6 {
            if self.0.iter().filter(|&&c| c as usize == f).count() != 16 {
                return Err(format!("colour {} does not appear 16 times", FACE_CHARS[f]));
            }
        }
        // Find the relabelling: the corner in slot DBL, read along its sticker order, must
        // become D, B, L in the same order.
        let cols: Vec<usize> = (0..3)
            .map(|i| at(CORNER_POS[DBL], g.corner_normals[DBL][i]))
            .collect();
        let want: Vec<usize> = (0..3)
            .map(|i| face_of_normal(g.corner_normals[DBL][i]))
            .collect();
        let mut relabel = None;
        for m in ROTATIONS.iter() {
            let mut map = [0u8; 6];
            for f in 0..6 {
                map[f] = face_of_normal(mat_apply(m, FACE_NORMAL[f])) as u8;
            }
            if (0..3).all(|i| map[cols[i]] as usize == want[i]) {
                relabel = Some(map);
                break;
            }
        }
        let map = relabel.ok_or("invalid DBL corner colours")?;
        let col = |pos: V3, n: V3| map[at(pos, n)] as usize;

        let mut cube = Cube::SOLVED;
        let mut seen_c = [false; 8];
        for s in 0..8 {
            let c: Vec<usize> = (0..3)
                .map(|i| col(CORNER_POS[s], g.corner_normals[s][i]))
                .collect();
            let mut found = false;
            for p in 0..8 {
                let pc: Vec<usize> = (0..3)
                    .map(|i| face_of_normal(g.corner_normals[p][i]))
                    .collect();
                for o in 0..3 {
                    // piece sticker i is on slot sticker (i + o) % 3
                    if (0..3).all(|i| c[(i + o) % 3] == pc[i]) {
                        if seen_c[p] {
                            return Err("duplicate corner".into());
                        }
                        seen_c[p] = true;
                        cube.cp[s] = p as u8;
                        cube.co[s] = o as u8;
                        found = true;
                    }
                }
            }
            if !found {
                return Err("invalid corner".into());
            }
        }
        if cube.co.iter().map(|&o| o as u32).sum::<u32>() % 3 != 0 {
            return Err("corner twist".into());
        }
        let mut seen_w = [false; 24];
        for s in 0..24 {
            let mut found = false;
            for p in 0..24 {
                let rm = rotation_mapping(g.wing_pos[p], g.wing_pos[s]);
                if g.wing_normals[p]
                    .iter()
                    .all(|&n| col(g.wing_pos[s], mat_apply(&rm, n)) == face_of_normal(n))
                {
                    if seen_w[p] {
                        return Err("duplicate wing".into());
                    }
                    seen_w[p] = true;
                    cube.wp[s] = p as u8;
                    found = true;
                }
            }
            if !found {
                return Err("invalid wing".into());
            }
        }
        let mut next = [0u8; 6];
        for s in 0..24 {
            let face = face_of_normal(normals(g.centre_pos[s])[0]);
            let c = col(g.centre_pos[s], FACE_NORMAL[face]);
            if next[c] >= 4 {
                return Err("too many centres of one colour".into());
            }
            cube.ct[s] = (c * 4) as u8 + next[c];
            next[c] += 1;
        }
        Ok((cube, map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turns_have_order_four() {
        for axis in 0..3u8 {
            for layers in 1..16u8 {
                let t = Turn { axis, layers, k: 1 };
                let mut c = Cube::SOLVED;
                for i in 0..4 {
                    if i > 0 {
                        assert_ne!(c, Cube::SOLVED);
                    }
                    c = c.apply_turn(t);
                }
                assert_eq!(c, Cube::SOLVED);
            }
        }
    }

    #[test]
    fn facelets_roundtrip() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let c = Cube::random(&mut rng);
            let f = c.to_facelets();
            let (c2, map) = f.to_cube().unwrap();
            assert_eq!(map, [0, 1, 2, 3, 4, 5]);
            assert_eq!(c2.cp, c.cp);
            assert_eq!(c2.co, c.co);
            assert_eq!(c2.wp, c.wp);
            assert_eq!(c2.centre_colours(), c.centre_colours());
        }
    }

    #[test]
    fn facelet_and_cubie_turns_agree() {
        let mut rng = rand::thread_rng();
        let c = Cube::random(&mut rng);
        for axis in 0..3u8 {
            for layers in 1..16u8 {
                for k in 1..4u8 {
                    let t = Turn { axis, layers, k };
                    assert_eq!(c.apply_turn(t).to_facelets(), c.to_facelets().apply_turn(t));
                }
            }
        }
    }

    #[test]
    fn rotated_cube_is_recognised() {
        let rot = Turn::rotation(R, 1);
        let f = Cube::SOLVED
            .to_facelets()
            .apply_turn(rot)
            .apply_turn(Turn::rotation(U, 1));
        assert!(f.is_solved());
        let (c, _) = f.to_cube().unwrap();
        assert!(c.is_solved());
    }

    #[test]
    fn sexy_move_order_six() {
        let seq = [
            Turn::face(R, 1, 1),
            Turn::face(U, 1, 1),
            Turn::face(R, 1, 3),
            Turn::face(U, 1, 3),
        ];
        let mut c = Cube::SOLVED;
        for _ in 0..6 {
            c = c.apply_turns(&seq);
        }
        assert_eq!(c, Cube::SOLVED);
    }
}
