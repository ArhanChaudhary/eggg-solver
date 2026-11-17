//! Move notation: parsing WCA and SiGN notation, and printing outer block turns.

use crate::cube::{F, FACE_CHARS, R, Turn, U};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Style {
    /// WCA notation: R, Rw, 3Rw.
    #[default]
    Wca,
    /// SiGN notation: R, r, 3r.
    Sign,
}

/// An outer block turn as written: `width` layers from `face`, turned clockwise `power` times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub face: u8,
    pub width: u8,
    pub power: u8,
}

impl Move {
    pub fn to_turn(self) -> Turn {
        Turn::face(self.face as usize, self.width, self.power)
    }

    pub fn format(self, style: Style) -> String {
        let f = FACE_CHARS[self.face as usize];
        let mut s = match (style, self.width) {
            (_, 1) => f.to_string(),
            (Style::Wca, 2) => format!("{f}w"),
            (Style::Wca, w) => format!("{w}{f}w"),
            (Style::Sign, 2) => f.to_ascii_lowercase().to_string(),
            (Style::Sign, w) => format!("{w}{}", f.to_ascii_lowercase()),
        };
        match self.power {
            2 => s.push('2'),
            3 => s.push('\''),
            _ => {}
        }
        s
    }
}

pub fn format_moves(moves: &[Move], style: Style) -> String {
    moves
        .iter()
        .map(|m| m.format(style))
        .collect::<Vec<_>>()
        .join(" ")
}

fn face_index(c: char) -> Option<usize> {
    FACE_CHARS.iter().position(|&f| f == c.to_ascii_uppercase())
}

/// Parse a sequence of moves in WCA or SiGN notation (both are accepted): outer turns
/// (`R`), wide turns (`Rw`, `r`, `3Rw`, `3r`), inner slices (`2R`), rotations (`x y z`).
pub fn parse_moves(s: &str) -> Result<Vec<Turn>, String> {
    let mut out = vec![];
    for tok in s.split_whitespace() {
        out.push(parse_move(tok)?);
    }
    Ok(out)
}

fn parse_move(tok: &str) -> Result<Turn, String> {
    let err = || format!("cannot parse move `{tok}`");
    let chars: Vec<char> = tok
        .chars()
        .map(|c| if c == '’' { '\'' } else { c })
        .collect();
    let mut i = 0;
    let mut prefix = None;
    while i < chars.len() && chars[i].is_ascii_digit() {
        prefix = Some(prefix.unwrap_or(0) * 10 + chars[i].to_digit(10).unwrap() as u8);
        i += 1;
    }
    let letter = *chars.get(i).ok_or_else(err)?;
    i += 1;
    let wide = chars.get(i) == Some(&'w');
    if wide {
        i += 1;
    }
    let mut power = 1u8;
    if i < chars.len() && chars[i].is_ascii_digit() {
        power = chars[i].to_digit(10).unwrap() as u8 % 4;
        i += 1;
    }
    if chars.get(i) == Some(&'\'') {
        power = (4 - power) % 4;
        i += 1;
    }
    if i != chars.len() {
        return Err(err());
    }
    if let Some(axis) = "xyz".find(letter) {
        if prefix.is_some() || wide {
            return Err(err());
        }
        return Ok(Turn::rotation([R, U, F][axis], power));
    }
    let face = face_index(letter).ok_or_else(err)?;
    let lower = letter.is_ascii_lowercase();
    let turn = match (lower, wide, prefix) {
        (false, false, None) => Turn::face(face, 1, power),
        (false, false, Some(n @ 1..=4)) => Turn::slice(face, n, power),
        (false, true, None) | (true, false, None) => Turn::face(face, 2, power),
        (_, _, Some(n @ 1..=4)) => Turn::face(face, n, power),
        _ => return Err(err()),
    };
    Ok(turn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format() {
        let t = parse_moves("R U' F2 Rw 3Rw' r 3r2 2R x y' z2 Uw2'").unwrap();
        assert_eq!(t.len(), 12);
        assert_eq!(t[3], t[5]);
        assert_eq!(t[4], Turn::face(R, 3, 3));
        let m = Move {
            face: R as u8,
            width: 2,
            power: 3,
        };
        assert_eq!(m.format(Style::Wca), "Rw'");
        assert_eq!(m.format(Style::Sign), "r'");
    }
}
