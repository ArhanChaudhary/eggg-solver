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
| `large` | h48 h6, joint phase 2, phase-1 move masks, wing paths up to 6 (+7 reduced) | 5.13 GB |
| `huge` | h48 h7, exact joint phase 2, phase-1 move masks, wing paths up to 7 | 9.5 GB |
| `xl` | h48 h9, exact joint phase 2, phase-1 move masks, wing paths up to 7 | 20.9 GB |
| `xxl` | h48 h11 (the largest), exact joint phase 2, phase-1 move masks, wing paths up to 7 | 66.5 GB |

Beyond the tables a run needs a few hundred MB, and that does not grow with the run time:
phase-1 endpoints become phase-2 candidates as they are found (only the few that can still
lead to a shorter solution are kept), and the move-mask batches run in chunks of bounded
size. One position with `huge` on 16 threads for 40 minutes: RSS 7.9-8.3 GB (before, it
grew to 34 GB, as each round held all its endpoints and the batches grew ~10x per phase-1
length). `EGGG_MEM=1` prints the driver's container sizes, the RSS and malloc's statistics
after each round; `EGGG_DEBUG=1` traces each round (phase-1 lengths with their nodes,
endpoints and kept candidates; per phase-2 level the candidates searched and finished, the
solutions found and kept and why it stopped; per phase-3 level the reductions searched; margin
changes; each improvement's phase lengths and the rounds that made it). A candidate takes 72
bytes (its phase-1 path inline, its cube recomputed when it is searched).

The tables: phase-1 centre staging (symmetry-reduced, 25 MB; `--p1-big`: 1.2 GB without the
reduction, faster lookups), phase-2 pairing permutation (60 MB; `--p2-joint`: the joint
pairing x U/D centres x PLL parity table, 1.05 GB, far fewer phase-2 nodes), phase-3 H48
(30 MB * 2^K for K edge-orientation bits, `--h48 K`). The three options override the tier.
The joint table stores distances mod 3 (2 bits), so a state's distance is a walk down the
table (~10 steps of several reads each); every phase-1 endpoint needs one or two such walks
to become a phase-2 candidate (about 1% are kept), 12-20% of the phase-1 time in long runs.
The huge and larger tiers hold its exact distances instead (`p2-joint-exact.tab`, 4 bits,
2.1 GB, same index; `--p2-exact true|false` overrides the tier): one read per bound. It is
made once from the mod-3 table, breadth-first (a neighbour of a state at distance d is at d + 1
exactly when its mod-3 value says so), in 58 s on 48 threads with a peak of 3.3 GB; the
distances go up to 14 and every entry agrees with the mod-3 table. Bounded distances of random
states: 1046 ns per walk, 15.5 ns per read (x86, one thread); phase 2 reads the same table
(its values mod 3; 1.3% more phase-2 CPU). The candidates are exactly the same. x86, 48
threads: making a candidate 2.5 -> 0.37 thread-us per endpoint frame, 12-14% -> 2.4-2.6% of
the phase-1 time, phase-1 nodes per second +14% (50 x 20 s, large tier with the exact table)
and +19% (10 positions x 600 s, huge); lengths unchanged (38.44 = 38.44 at 20 s; 38.3 = 38.3
at 600 s), as expected for a uniform speed-up in the logarithmic regime of long runs (+19% is a
quarter of a doubling of the time).
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
table (`src/wingpath.rs`, built at start-up in a second or two: 9 MB up to distance 5,
0.42 GB up to 6 with distance 7 reduced by symmetry (the large tier), 1.8 GB up to 7 (huge
and larger); 15.2 GB up to 8, built once in about a minute with a peak of 15.0 GB and saved
as `p1-wingb-k8s6.tab`; `--p1-wing K` overrides it, `EGGG_P1_WINGS` sets the slack-1 lists'
distance (5, or 6 with K >= 7), `EGGG_P1_WINGS2` the slack-2 lists' (4; -1: none),
`EGGG_P1_WINGR=0/1` the next distance stored once per symmetry class (on with K <= 6)) tests
a node's tight and slack-1 children before they are queued: same endpoints, 9x fewer mask
reads, p1check 2.55x faster on the M2 (1 thread, 3.59 s -> 1.41 s). 50 random positions on the M2 (seed 6060), up to distance 7:
38.94 -> 38.74 moves after 20 s with 8 threads (8 shorter, none longer), 39.70 -> 39.50
with one thread after 10 s. The frame of a (frame, preimage) pair does not matter (in every
frame the high-piece parity condition is the same: an even number of the dedges' first
wings in the preimage), so the lists hold preimages only, bit-sliced eight per 24-byte plane
and tested eight at a time with one gather by the wings' slots; the tests are split in three
passes over a batch chunk (keys and slots prefetched, lists located and prefetched, lists
read); children with two moves of slack are tested too (slack-2 lists up to distance 4, 25
MB), with the tight lists to 7 the slack-1 lists go to 6 (0.46 GB), and the children's
liveness bits are evaluated for all moves at once. Same endpoints;
p1check on the M2 (1 thread): 1.61 -> 1.00 s with the large tier's table (-38%), 1.45 ->
0.80 s with the huge tier's (-45%); 50 positions on the M2 (large tier, 8 threads, 20 s):
38.78 -> 38.66 moves (4 shorter, none longer; 39.04 -> 38.94 at 10 s), one thread after 10 s
unchanged (39.52). The reduced lists' tests of the tight children of nodes with slack go
through the same three passes, and the class search is a branch-free, unrolled loop over the
six symmetries: p1check deep with the large tier's table 1.00 -> 0.92 s on the M2 (1
thread), phase-1 nodes per second +6% (M2, 8 threads) and +8.5% (x86, 48 threads).
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
at most 14 moves at first, 13 after 3 s and 12 after 6 s. Each deeper level has ~13x the
reductions and each is ~13x less likely to have a short enough phase 3, so every level adds
about as many chances, and long runs keep improving. In long runs phases 2 and 3 run out of
work: after a minute on 48 threads each round's phase 2 finishes every level up to the cap in
a few hundredths of a second (0.1-0.5% of the time), the reduction budget no longer binds and
phase 3 is never cut short, while phase 1 spends minutes on its last length (17 moves, about
20x the work of length 16). So from 20 s on the cap goes one level deeper (phase 3 of at most
11, then 10 moves) whenever 16 times the share of phases 2 and 3 in the latest rounds is below
a quarter; candidates already searched keep their next level for that, and phase 2 then runs
without a reduction budget (but at most as long as the round's phase 1). Measured per level
(`EGGG_DEBUG=1`, 300 s runs): the phase-2 solutions grow ~13x per level and their expected
improvements (solutions x P(optimal 3x3 short enough)) are equal to within 5%, so the two
deeper levels add ~40% more chance per round for ~2% of the time (`EGGG_ADAPT=inf
EGGG_P2_FREE=inf EGGG_P2_TIME=inf` turns this off). The rounds are timed against the
limit: once a solution exists, each round's phase 1 stops early enough (at 1/1.3 of the time
left) for that round's phase 2 and 3 to run before the limit, rounds last at most a quarter
of the time elapsed so improvements arrive steadily, and deep phase-2 searches are split into
tasks over the threads. 48 threads (x86 server), 50 random positions (seed 6060), current
version: 38.56 / 38.54 / 38.52 / 38.50 moves on average after 10 s with the large / huge /
xl / xxl tiers (the larger tables mainly help in the first second: 39.66 / 39.46 / 39.34 /
39.32 after 0.5 s; the 15 GB depth-8 wing-path table, `--p1-wing 8`, adds nothing at 48
threads); 50 other positions (seed 3030): 38.76 after 2 s with the huge tier; 20 positions
(seed 6060, huge) for 5 minutes: 38.50 / 38.45 / 38.40 / 38.30 after 10 s / 1 / 2 / 5 minutes
without the deeper caps, 38.55 / 38.50 / 38.40 / 38.25 with them (one position shorter at 5
minutes, none longer; 4 of the 6 improvements after 10 s came from the two deeper levels).
(Before the
masks, the round cut, the faster phase 2 and the wing-path tables, 2026-11-17: 39.7 / 39.5
after 2 s, 38.8 / 39.1 after 60 s on two sets of 20 positions.)

## Development

- `cargo test --release` (end-to-end tests use `./tables`, generating them if missing).
- Benchmarks (random positions; best length after a time limit, multi- and single-threaded):
  `scripts/x86-build.sh LABEL REF` + `scripts/x86-bench.sh NAME...` on this machine,
  `scripts/m2-build.sh` + `scripts/m2-bench.sh` (and `scripts/m2-long.sh` for paired long
  runs) on the ARM laptop (each under an exclusive benchmark lock), then `scripts/compare.py A B` for a before/after table and
  `scripts/paired.py A.json B.json` for a per-position paired comparison (runs of one build
  reproduce per-position lengths, so small differences are measurable).
- Long runs (`scripts/erate.py`): a 20-position run of 10 minutes improves only ~5 times after
  10 s, so paired final lengths cannot resolve effects under ~2x. Each phase-2 solution of
  total length L12 found at best length B is a draw that wins if its optimal phase 3 has at
  most B - 1 - L12 moves, distributed like the distance of a random 3x3 position (P(<= 17) =
  0.305, P(<= 18) = 0.965; the table in the script, from Rokicki et al. / cube20.org), so the
  run's expected improvements are E = sum over rounds of 1 - exp(-(sum of solutions x
  P(3x3 distance <= B - 1 - L12))) (several successes in one round make one improvement;
  `--raw`: without that). `EGGG_DEBUG=1 eggg-solver bench ... 2> X.trace` writes the trace (its
  `P2 start`, `P2LEVEL <L12>: ... seen <N> best <B>` and `IMPROVE` lines are used;
  `scripts/x86-long.sh` writes `$EGGG_TRACE_DIR/NAME.trace` when that is set);
  `scripts/erate.py X.trace` prints E and the observed improvements per position and time
  window, `scripts/erate.py --pair A.trace B.trace` the E each run gathered while both had
  the same best (same positions, same wall time: a run that improves sooner faces a harder
  target afterwards), the ratio B/A with a bootstrap interval over positions, and the
  geometric mean of the per-position ratios (`--from S`, default 10 s, `--to S`). Two master
  runs of 20 x 600 s (x86): B/A 1.000 (90% interval 0.95-1.05), where their final lengths
  could not resolve 2x. In 20-s runs it is much noisier (±15-25%: few rounds at a settled
  best); use the paired lengths there. E is calibrated loosely: the improvements observed
  after 10 s in six long runs (300-600 s, 32 improvements) were 0.69 of it (0.5-1.1 per run;
  the solutions of one candidate are correlated), between 2 and 20 s in two 50 x 20 s runs
  1.04 (59 improvements).
- Profile-guided builds (LLVM PGO): `scripts/pgo-build.sh x86|m2 LABEL REF` (an instrumented
  build, a one-minute training run on the machine the binary is for, `llvm-profdata` from the
  toolchain's `llvm-tools` component, and the release build with the profile; per commit).
  On the M2 phase 1 runs ~8% (8 threads) / ~6% (1 thread) more nodes per second than the
  plain build; the final lengths of the 50-position suites did not change.
- `examples/` holds deterministic checks used during optimisation: `p1check` (phase-1
  endpoint hash and node count), `p2jcheck` (joint-table admissibility), `p2cost` (phase-2
  nodes and CPU time per solution, by depth above the bound), `p2exact` (bounded joint
  distances read from the exact table against walks down the mod-3 one, and their times;
  makes the exact table if missing), `p3check`,
  `p3nodes` and `p3tasks` (phase-3 bounds, node counts and task splitting).
