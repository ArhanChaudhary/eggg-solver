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
# Ten random positions, 5 seconds each, solutions only (scrambles can also come one per
# line on stdin):
target/release/eggg-solver solve --random 10 -t 5 -q
# Also print every other solution as short as the best so far (equally short alternatives):
target/release/eggg-solver solve "..." -t 30 --ties
# Generate the tables up front, or benchmark:
target/release/eggg-solver gen
target/release/eggg-solver bench -N 10 -t 5
```

Each improvement is printed as it is found (`time, length, phase lengths, moves`), then the
best solution. Without `-t`, `solve` searches until it is stopped (Ctrl-C) or `-l` is reached;
`bench` defaults to 5 seconds per position. With `--ties`, solutions as long as the best so far are printed too (one per
reduction, each move sequence once; phase 3 then also searches the best length, so they
mostly appear in longer runs). The time limit is soft until a first solution exists (`--hard-limit` to cap it).

### Tables and memory

Tables are generated on first use and saved in `--tables DIR` (default `$EGGG_TABLES`, else
`./tables`); the large ones take a few minutes, once. `--tier` picks the set by memory use;
the default, `auto`, takes the largest tier that fits in a third of the physical memory
(`xl` and `xxl` only once their phase-3 table exists: generate it once with
`eggg-solver gen --tier xl`, ~20 minutes):

| tier | tables | memory |
|---|---|---:|
| `tiny` | h48 h0 | 0.16 GB |
| `small` | h48 h3 | 0.35 GB |
| `medium` | h48 h5, joint phase 2 | 2.3 GB |
| `large` | h48 h6, joint phase 2, phase-1 move masks, wing paths up to 6 | 4.85 GB |
| `huge` | h48 h7, joint phase 2, phase-1 move masks, wing paths up to 7 | 7.9 GB |
| `xl` | h48 h9, joint phase 2, phase-1 move masks, wing paths up to 7 | 19.3 GB |
| `xxl` | h48 h11 (the largest), joint phase 2, phase-1 move masks, wing paths up to 7 | 64.9 GB |

The tables: phase-1 centre staging (symmetry-reduced, 25 MB; `--p1-big`: 1.2 GB without the
reduction, faster lookups), phase-2 pairing permutation (60 MB; `--p2-joint`: the joint
pairing x U/D centres x PLL parity table, 1.05 GB, far fewer phase-2 nodes), phase-3 H48
(30 MB * 2^K for K edge-orientation bits, `--h48 K`). The three options override the tier.
The larger phase-3 tables matter most with few threads, where phase 3 is a large share of
the time (one thread: first solution after 6.2 s with `huge`, 1.4 s with `xl`, 0.5 s with
`xxl`, on average); with many threads phase 1 dominates.

Phase-1 move masks (same phase-1 endpoints, fewer table reads; see `src/p1mask.rs`): the
reduced move-mask table (`p1-mask-live.tab`, 1.6 GB, generated in ~15 s; on by default
in the large and larger tiers, `--p1-mask none` to turn it off) stores, per staging state,
the moves that decrease the staging distance: deep in the tree almost every node is at
distance = moves left, and then one read of its mask replaces ~20 reads of its children; such
nodes are expanded in batches with the masks prefetched a round ahead (10x fewer reads in
all; phase 1 5x faster on the M2; in 60 s runs on the M2, 8 threads, 20 positions: 39.70 ->
39.30 moves, and the masked solver reaches at 10 s what it reached at 60 s without them). Its
entries also mark the wing parities with which a node at distance = moves left can still end
at an endpoint (over half of these nodes cannot, and are dropped), and hold the moves that
keep the distance: a node with slack gets all its children's distances from its own entry,
so the search needs no large staging table. The entries also say, per move, whether the child via the
move can still reach an endpoint when it is at distance = moves left (its parity viability,
and a decreasing move that the canonical order allows after the move; Dw, Lw and Bw share the
bits of Uw, Rw and Fw, which makes room): such children are neither queued nor tested (27%
fewer batch nodes, phase 1 1.2x faster on the M2, same endpoints). A per-node pass
over the masked search (parent-pointer paths, cheaper leaf tests, byte-lookup pext on ARM)
made it another 3x faster with the same endpoints, and the parity and equal-distance bits
2x more. A node two moves from the end is only searched if its wings can still be made
pairable by the last two moves (the last one a wide quarter turn, or with staged centres an
outer quarter turn of the special axis): a test of its wing positions against the 12 to 51
slot sets per frame that those move pairs take to the low slots, bit-sliced into one word
per frame (`src/wing2.rs`, no table). About 90% of these nodes, and of the last-move tests
below them, go (same endpoints; p1check 1.25x faster on the M2, 1 thread). Stronger, with
the centres: a node at staging distance = moves left (or one less) can only end by one of
its few staging paths (2.3 per node 3 moves from the end, 4.4 at 6; one keeping move
allowed with slack), so per centre state (the partition of the centres into colour-pair
sets) the distinct (frame, preimage of the frame's low slots under the path) pairs decide
whether its wings can still be paired: 3 to 8 pairs per state, and 99.5% to 99.8% of the
nodes 3 to 8 moves from the end fail them all (the two-move test passes 5%). The wing-path
table (`src/wingpath.rs`, built at start-up in about a second: 9 MB up to distance 5,
144 MB up to 6 (the large tier), 1.24 GB up to 7 (huge and larger); 14.6 GB up to 8, built
once in about a minute with a peak of 14.7 GB and saved as `p1-wing-k8s5.tab`; `--p1-wing K`
overrides it, `EGGG_P1_WINGS` sets the slack-1 lists' distance, `EGGG_P1_WINGR=1` adds the
next distance stored once per symmetry class) tests a node's tight and slack-1 children
before they are queued: same endpoints, 9x fewer mask reads, p1check 2.55x faster on the M2
(1 thread, 3.59 s -> 1.41 s). 50 random positions on the M2 (seed 6060), up to distance 7:
38.94 -> 38.74 moves after 20 s with 8 threads (8 shorter, none longer), 39.70 -> 39.50
with one thread after 10 s.
On the M2 (8 threads), 20 random positions (seed 6060), current version: 39.45
moves on average after 10 s, 39.10 after 30 s, 38.80 after 60 s (before the parity and
equal-distance bits, the round cut and the cheaper candidates: 39.65 / 39.30 / 39.15, and
38.80 only after 2 minutes, 38.60 after 10 minutes). `--p1-mask big` is the same without the symmetry reduction (18.9 GB).
Experimental (see `src/p1pat.rs`): `--p1-pat small|big` (0.9 / 41 GB) runs one phase-1 search
per colour pair, pruned by the distance to staged centres with that pair in a pattern phase
2 can solve (0.53x reads, 1.4x faster on the M2); with `--p1-mask` too, the pair searches use
masks of that distance (17.5 GB).

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
   permutation table or the much tighter joint table, which is read twice per state: for the
   U/D centres, and for the F/B centres through the rotation x (which swaps the two axes and
   keeps the phase-2 moves), the larger of the two being the bound.
3. **Phase 3**: solve the reduced 3x3x3 in one IDA* search with H48 pruning tables (as in
   nissy's H48 solver and egg-solver), from both ends of the solution at once.

The driver collects many phase-1 endpoints and reductions and searches phase 3 level by
level on the total length over all of them, so each solution found is the shortest the pool
allows; the pool grows while time remains. Once a solution exists, each new phase-1 length searches
its jobs (3-move prefixes) best first, by their kept-candidate yield per node at the previous
length, so the length the time limit cuts short has searched its most productive subtrees. Once a solution exists, phase 2 only produces
reductions that a short enough phase 3 could still turn into a shorter solution: phase 3 of
at most 14 moves at first, 13 after 3 s and 12 after 6 s. Each deeper level has ~16x the
reductions and each is ~13x less likely to have a short enough phase 3, so every level adds
about as many chances, and long runs keep improving. The rounds are timed against the
limit: once a solution exists, each round's phase 1 stops early enough (at 1/1.3 of the time
left) for that round's phase 2 and 3 to run before the limit, rounds last at most a quarter
of the time elapsed so improvements arrive steadily, and deep phase-2 searches are split into
tasks over the threads. 48 threads, two sets of 20 random
positions: 39.7 / 39.5 moves on average after 2 s, 39.0 / 39.1 after 30 s, 38.8 / 39.1
after 60 s.

## Development

- `cargo test --release` (end-to-end tests use `./tables`, generating them if missing).
- Benchmarks (random positions; best length after a time limit, multi- and single-threaded):
  `scripts/x86-build.sh LABEL REF` + `scripts/x86-bench.sh NAME...` on this machine,
  `scripts/m2-build.sh` + `scripts/m2-bench.sh` (and `scripts/m2-long.sh` for paired long
  runs) on the ARM laptop (each under an exclusive benchmark lock), then `scripts/compare.py A B` for a before/after table and
  `scripts/paired.py A.json B.json` for a per-position paired comparison (runs of one build
  reproduce per-position lengths, so small differences are measurable).
- Profile-guided builds (LLVM PGO): `scripts/pgo-build.sh x86|m2 LABEL REF` (an instrumented
  build, a one-minute training run on the machine the binary is for, `llvm-profdata` from the
  toolchain's `llvm-tools` component, and the release build with the profile; per commit).
  On the M2 phase 1 runs ~8% (8 threads) / ~6% (1 thread) more nodes per second than the
  plain build; the final lengths of the 50-position suites did not change.
- `examples/` holds deterministic checks used during optimisation: `p1check` (phase-1
  endpoint hash and node count), `p2jcheck` (joint-table admissibility), `p2cost` (phase-2
  nodes and CPU time per solution, by depth above the bound), `p3check`,
  `p3nodes` and `p3tasks` (phase-3 bounds, node counts and task splitting).
