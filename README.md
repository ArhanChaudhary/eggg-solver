# eggg-solver

A fast three-phase 4x4x4 Rubik's cube solver in Rust. It looks for **short solutions in the
outer block turn metric (OBTM), quickly**, and keeps improving its solution until a time limit.
Output in WCA (`Rw`) or SiGN (`r`) notation.

## Usage

```sh
cargo build --release
# Solve a scramble (WCA or SiGN notation, rotations and inner slices allowed) for 5 seconds:
target/release/eggg-solver solve "Rw U2 x Rw2 F' Bw D Lw' 2R B2 u'" -t 5
# A 96-character facelet string (faces U R F D L B, each row by row):
target/release/eggg-solver solve UUUU...BBBB
# SiGN output, all cores, stop at the first solution of at most 40 moves:
target/release/eggg-solver solve "..." -n sign -l 40
# Ten random positions, solutions only (scrambles can also come one per line on stdin):
target/release/eggg-solver solve --random 10 -q
# Generate the tables up front, or benchmark:
target/release/eggg-solver gen
target/release/eggg-solver bench -N 10 -t 5
```

Each improvement is printed as it is found (`time, length, phase lengths, moves`), then the
best solution. The time limit is soft until a first solution exists (`--hard-limit` to cap it).

### Tables and memory

Tables are generated on first use and saved in `--tables DIR` (default `$EGGG_TABLES`, else
`./tables`); the large ones take a few minutes, once. `--tier` picks the set by memory use;
the default, `auto`, takes the largest tier that fits in a third of the physical memory:

| tier | tables | memory |
|---|---|---:|
| `tiny` | h48 h0 | 0.16 GB |
| `small` | h48 h3 | 0.35 GB |
| `medium` | h48 h5, joint phase 2 | 2.3 GB |
| `large` | h48 h6, joint phase 2, large phase 1 | 4.2 GB |
| `huge` | h48 h7, joint phase 2, large phase 1 | 6.1 GB |

The tables: phase-1 centre staging (symmetry-reduced, 25 MB; `--p1-big`: 1.2 GB without the
reduction, faster lookups), phase-2 pairing permutation (60 MB; `--p2-joint`: the joint
pairing x U/D centres x PLL parity table, 1.05 GB, far fewer phase-2 nodes), phase-3 H48
(30 MB * 2^K for K edge-orientation bits, `--h48 K`). The three options override the tier.

## How it works

The cube is solved in a fixed frame with 27 moves (U R F D L B Uw Rw Fw and their powers);
Dw, Lw and Bw are never needed (Dw is Uw plus a rotation) and the cube may end solved in any
orientation, so the output needs no rotations.

1. **Phase 1** (Tsai/TPR phases 1+2, merged): put every centre on a face of its colour pair's
   axis, the centres of one axis into a pattern phase 2 can finish, make the wing permutation
   even and the wings *pairable* (the 12 "low" wing slots of the phase-2 frame hold one wing of
   each dedge, an even number of them "high"). One search tree serves all 18 combinations of
   frame (which axis phase 2 turns only by half turns) and colour-to-axis assignment. Guided by
   an exact staging table (reduced by the 48 symmetries, or the large unreduced one); the
   search is split into jobs that hand back their unstarted subtrees when stopped, so threads
   never wait for one another.
2. **Phase 2** (TPR phase 3): pair the dedges and solve the centres with
   U D F B R2 L2 Uw2 Rw2 Fw2, guided by an exact centre table and either the pairing
   permutation table or the much tighter joint table.
3. **Phase 3**: solve the reduced 3x3x3 in one IDA* search with H48 pruning tables (as in
   nissy's H48 solver and egg-solver), from both ends of the solution at once.

The driver collects many phase-1 endpoints and reductions and searches phase 3 level by
level on the total length over all of them, so each solution found is the shortest the pool
allows; the pool grows while time remains, and once a solution exists phase 2 only produces
reductions that a phase 3 of at most 14 moves would turn into a shorter one.

## Development

- `cargo test --release` (end-to-end tests use `./tables`, generating them if missing).
- Benchmarks (random positions; best length after a time limit, multi- and single-threaded):
  `scripts/x86-build.sh LABEL REF` + `scripts/x86-bench.sh NAME...` on this machine,
  `scripts/m2-build.sh` + `scripts/m2-bench.sh` on the ARM laptop (each under an exclusive
  benchmark lock), then `scripts/compare.py A B` for a before/after table and
  `scripts/paired.py A.json B.json` for a per-position paired comparison (runs of one build
  reproduce per-position lengths, so small differences are measurable).
- `examples/` holds deterministic checks used during optimisation: `p1check` (phase-1
  endpoint hash and node count), `p2jcheck` (joint-table admissibility), `p3check`,
  `p3nodes` and `p3tasks` (phase-3 bounds, node counts and task splitting).
