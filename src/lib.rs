//! eggg-solver: a fast three-phase 4x4x4 solver in the outer block turn metric.

pub mod c3simd;
pub mod comb;
pub mod cube;
pub mod cube3;
pub mod h48;
pub mod moves;
pub mod notation;
pub mod par;
pub mod phase1;
pub mod phase2;
pub mod phase3;
pub mod simd;
pub mod solver;
pub mod table;
