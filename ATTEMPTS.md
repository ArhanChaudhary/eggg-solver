# Optimisation log: everything tried so far

Status as of 2026-11-23. ✅ merged into master, ❌ tried and not merged (measured), ⏳ in
progress, ⏸ waiting for the 48-thread x86 server (busy with another project since
2026-11-17). Numbers are mean solution lengths (moves) on random positions unless stated;
"M2" is the 8-core ARM laptop, "x86" the 48-core server. Branch names in `code`; every
branch, merged or not, is on GitHub.

**Where things stand:** at 20 s on the M2 (8 threads, 50 positions) the solver averages
**38.66** moves; on 2026-11-18 it needed about 2 minutes on the same machine for 38.80.
Phase 1 is still most of the run time, so most recent work targets it.

## Phase 1 (staged centres + pairable wings)

| | Attempt | Summary |
|---|---|---|
| ✅ | Last-move wing test, prefetch pipeline, stabiliser-free lookups | Wing conditions tested from the parent before the leaf is built, with table reads prefetched. The first big phase-1 speedup. |
| ✅ | Split jobs on demand | Running jobs hand back unstarted subtrees, so no thread waits for the slowest one. |
| ✅ | Large unreduced staging table (1.2 GB), SIMD wing tests, huge pages | Faster lookups and tests. The masked tiers no longer load the large table. |
| ✅ | Perf-guided per-node pass | Exact micro-optimisations in all three phases, found with perf. |
| ✅ | Move-mask table (1.6 GB) | Per staging state, the moves that lower the distance, with nodes expanded in prefetched batches. Phase 1 5x faster on the M2; 60 s: 39.70 → 39.30. |
| ✅ | Masked per-node pass | Parent-pointer paths, cheaper leaf tests. Another 3x; 60 s: 39.30 → 39.15. |
| ✅ | Parity-viability bits + equal-distance masks | Exact rule dropping the 57% of batch nodes whose wing parity can't reach an endpoint, plus one read per slack node. Phase 1 2x; 60 s: 39.15 → 39.00. |
| ✅ | Wing check two moves before the end (`opt/p1-wing2`) | A table-free ~70-operation filter before a node is queued. Phase 1 1.24x; 30 s: 39.10 → 38.95. |
| ✅ | Job ordering by previous-level yield (`opt/p1-order`) | Each new length's 3-move jobs run best-first. 60 s: 38.75 → 38.65. |
| ✅ | Child "live" bits in the mask entries (`opt/p1-live`) | Precomputed bits skip the 29% of children that could never progress. +19% phase-1 throughput (8 threads), +24% (1 thread); lengths about equal. |
| ✅ | **Wing-path table** (`opt/p1-wing3`) | Per centre state, the wing slot sets its few staging paths need, so only 0.2–0.5% of tight nodes pass (5.2% before). Phase-1 check 2.55x faster; 50 positions × 20 s: **38.94 → 38.74** (8 shorter, 0 longer). |
| ✅ | Phase boundary hand-over | Phase 1's trailing same-axis moves go to phase 2, so reductions are no longer missed at the phase-1/phase-2 boundary. |
| ✅ | Cheaper, stronger wing-path tests (`opt/p1-wingpath2`) | Key cached from the prefetch pass, bit-sliced preimage lists, lists for nodes with two spare moves, reduced next-layer lists, branch-free liveness: phase-1 check 38–45% faster, same endpoints; 50 × 20 s 38.78 → 38.66 (4 shorter, 0 longer). |
| ❌ | Wide 64-preimage slices (`opt/p1-wingpath2-wide`) | 3–10% slower than the 8-per-plane slicing. |
| ⏳ | Prefetched slack-node reduced-list tests, cheaper class search, longer slack-list prefetch (`opt/p1-wingpath3`) | 21% of the reduced-list tests (tight children of nodes with slack, 8 moves from the end) ran unprefetched, class search and table reads in series; they now go through the three-pass pipeline. The class search (least key over six symmetries, ~190 instructions) is an unrolled branch-free loop without the identity's lookups. Same endpoints; p1check deep, large tier, 1 thread: −8.5% (M2), −8% (x86); phase-1 nodes/s +6% (M2, 8 threads), +8.5% (x86, 48 threads). x86 48 threads, large tier, 50 positions: 38.56 → 38.56 at 10 s, 38.44 → 38.42 at 20 s (neutral). The M2 suite is still to run. |
| ❌ | Per-thread cache of the reduced classes (`opt/p1-wingpath3-cache`) | 55–67% hits, but after the cheaper class search it gains 0.6% at best (hit/miss branches mispredict). |
| ❌ | Wide blocks for long lists, again (`opt/p1-wingpath3-wide`) | 64 preimages per block in the space of eight planes: p1check 3% slower, neutral at 8 threads, though the 8-thread solver scans many long slack-2 lists. |
| ❌ | Two planes per iteration (`opt/p1-wingpath3-twoplanes`) | No loop exit that depends on the list length for short lists: branch misses −4%, but 3% slower. |
| ❌ | Lanes (`opt/p1-lanes`) | Several jobs interleaved per thread: same endpoints, slower. |
| ❌ | Parity rule + staged-parent shortcut (`opt/p1-shortcuts`) | Exact, but only 0.7% fewer nodes and slower overall. |
| ❌ | Slices (`opt/slices`) | Cheap endpoints with an H-move tail searched first. Helped short runs, hurt long ones. |
| ❌ | Inverse-scramble root (`opt/inverse`) | A second root on the inverse scramble. Invalid on 4x4: identical centres make the inverse a different coset. |
| ❌ | Hand-back down to 7-move subtrees (`opt/p1-handback`) | Fixed the round-cut overshoot but cost 35% of phase-1 throughput. |
| ⏸ | Hand-back at any depth (`opt/p1-spill`, `opt/p1-spill2`) | Neutral at 8 threads, +6% at 1 thread. Held for a 48-thread test, where more threads wait at a cut. |
| ❌ | Candidates made inside the phase-1 workers (`opt/p1-stream`) | Neutral. |
| ❌ | Phase-2-aware phase 1 (`method/p1-h2` study) | 99.94% of phase-1 nodes have no endpoint below them, so even a perfect phase-2 bound removes 0.06%. Staging × pairing tables can't fit in 20–60 GB either. |
| ❌ | SIMD across batch nodes (`proto/p1-simd`) | Only the wing check vectorises (no gather for the table lookups), and a scalar rewrite matched it. |
| ❌ | Scalar wing-check rewrite (`proto/p1-w2packed`) | 7% in a microbenchmark but only +2% in the solver. |
| ❌ | Successor table (`opt/p1-succ`, `opt/p1-succ-live`) | Children's table indices stored per entry (1.6–3.2 GB). Net zero at first; 2–3% slower after the live bits. |
| ❌ | Retuning + branch-free liveness test (`opt/p1-tune`) | Constants were already optimal; the branch-free test was 3.4% slower. |
| ❌ | Cheaper leaf staging test (`prof/leaf`) | −0.6%; not worth a branch. |
| ❌ | Centre-blind wing table 3 moves out | Prunes only 54% of 3-left nodes, at about the cost it saves. Led to the wing-path table instead. |
| ❌ | Pattern tables (`--p1-pat`, opt-in) | Per-colour-pair searches: 1.4x fewer reads, superseded by the masks. |
| ❌ | Finer job ordering (`opt/p1-order2`) | 4- and 5-move jobs ranked by hierarchical yield: at most +3–6% kept candidates (oracle +16–37%), neutral in length (38.96 vs 38.94). |
| ✅ | Wing-path depth 8 (`opt/wing-mem`, `--p1-wing 8`) | The build used ~2x the final size (OOM on a 32 GB machine); now the table is 15.2 GB, built in ~40 s and saved to disk. Opt-in only: at 48 threads (50 × 10 s) huge/xl/xxl with depth 8 match depth 7 to within one position. |
| ⏸ | Unreduced 8-byte mask table (~38 GB) | No symmetry handling at all; estimated 1.2–1.3x. Needs the x86 server's memory. |
| ⏸ | 17.5 GB pair-mask table, 18.9 GB large mask table | Existing opt-in options, never benchmarked at full size. |

## Phase 2 (pair the dedges, solve centres)

| | Attempt | Summary |
|---|---|---|
| ✅ | Joint phase-2 table (1.05 GB) | Centre pattern × pairing distance: far fewer phase-2 nodes. |
| ✅ | Level truncation; streaming solutions | Bounded work and memory per level. |
| ✅ | Cheaper candidate creation | Cached standardisation and bounded, interleaved joint-table walks: 10 → 2.2 µs per candidate, same candidates. |
| ✅ | Split deep searches into tasks (`opt/p2-split`) | Phase-2 idle time 3.3 → 0.5 s; the final length is reached sooner. |
| ✅ | F/B view of the joint table + prefetch (`opt/p2-fb`) | The same table read from a second side: 4.5x less CPU per solution. 30 s: 39.10 → 38.95. |
| ✅ | Vector indexing of both views (`opt/p2-vindex`) | −31% phase-2 CPU; lengths unchanged (phase 2 is now ~4% of the time). |
| ❌ | Larger joint tables (6.35 / 18.4 GB) + deeper margins (`method/p2-big`, `int/p2-big`) | No gain, even in 300 s runs. |
| ❌ | Cheaper per-solution bookkeeping (`exp/margins`) | Neutral: the searches, not the bookkeeping, dominate. |
| ❌ | 4x more reductions per round | Phase 2's share rose from 12.6% to 19.2% with identical final lengths. |
| ❌ | 8x / unlimited phase-2 budget, re-tested after the phase-1 speedups | 38.80 / 38.90 vs 38.78 at 20 s: the per-round budget is not what limits length. |

## Phase 3 (optimal 3x3 with H48 tables)

| | Attempt | Summary |
|---|---|---|
| ✅ | Task split, mod-3 decode lookup, pext without BMI2 on ARM | Better thread use and cheaper lookups. |
| ✅ | xl (h9, 15.6 GB) and xxl (h11, 62 GB) tiers | Single-thread first solutions 6.2 → 1.4 → 0.5 s. No length gain at 48 threads, where phase 1 dominates. |
| ❌ | Learned bounds, carried quarter-turn masks, side choice (`opt/p3-search`) | Ideas from egg-solver; slower here. |
| ❌ | Phase-3 lanes (`opt/p3-lanes`) | Neutral: phase 3 is a small share of the time. |
| ❌ | Quick first solution (`opt/quick-first`) | Solving the most promising reduction outright traded length for latency. |
| ❌ | Wide-half-turn endgames (`method/h-endgame`) | A last phase allowing wide half turns: shortcuts exist but in ~1e-5 of the pool. |
| ❌ | Corner-first phase 3 with a colour-complement joint table (`opt/p3-corner-first`) | Unfinished prototype; superseded. |

## Driver (rounds, pools, time)

| | Attempt | Summary |
|---|---|---|
| ✅ | Parallel candidate creation, balanced driver | Better thread use between phases. |
| ✅ | Deep phase-2 caps (margins 15/14/13), no 64-solution limit | Every deeper phase-2 level adds about as much chance as the one above. x86 60 s: 39.2 → 38.8. |
| ✅ | Bounded memory in long runs | Drops hopeless candidates and reductions; fixed a 74 GB blow-up in the 1-hour run. |
| ✅ | Phase 3 gets 3x the budget until the first solution | First solution 2.67 → 1.24 s (M2 mean). |
| ✅ | Round cut (`opt/round-cut`) | The round running at the deadline is searched instead of thrown away (19–35% of the time before). 60 s: 39.15 → 38.95; overrun 1.1 → 0.04 s. |
| ✅ | Smaller rounds after the first solution (`opt/round-small`) | Improvements arrive sooner; final lengths unchanged, also at 2 s on the M2 and at 2 s with 48 threads (39.18 vs 39.20, 50 positions). |
| ❌ | Check-first scheduler (`opt/scheduler`) | Always doing the cheapest work first; its phase-3 checks cost too much. |
| ❌ | Phase-3 budget / round growth tuning (`opt/tune`), long-run caps (`opt/long-tune`), margin schedules | The existing constants were already at the optimum. |

## Tables, tooling, infrastructure

| | Attempt | Summary |
|---|---|---|
| ✅ | Tier sweep at 48 threads (x86, 50 × 10 s) | large / huge / xl / xxl: 38.56 / 38.54 / 38.52 / 38.50 at 10 s; the bigger tables mainly help in the first second (39.66 → 39.32 at 0.5 s). |
| ✅ | Tiers (`--tier`, auto by memory) | tiny / small / medium / large / huge / xl / xxl = 0.16 / 0.35 / 2.3 / 4.85 / 7.9 / 19.3 / 64.9 GB. Mix and match with `--h48`, `--p2-joint`, `--p1-mask`, `--p1-wing`. |
| ✅ | Bounded memory in long runs (`fix/long-run-memory`) | Rounds kept every phase-1 endpoint until the round ended, and deep batches grew 10x per length: 8 → 34 GB in 40 minutes. Candidates are now made as endpoints are found, batches run in chunks: 7.9–8.3 GB over 90 minutes, same lengths (8-thread phase-1 rate −3.6%). |
| ✅ | `solve --ties`; no time limit by default | Print every solution as short as the best so far; `solve` runs until stopped or `-l` (bench keeps 5 s). |
| ✅ | Concurrent table loading | A single `solve` on the M2: 7.35 → 5.64 s, every table still checksummed. |
| ✅ | PGO build script (`scripts/pgo-build.sh`) | +6.5–8% phase-1 throughput with a per-machine, per-commit training run. Optional. |
| ✅ | Fixes | Concurrent table saves; M2 build scripts that could compile another agent's code or copy a stale binary; the renamed M2 SSH key. |
| ✅ | Benchmarking | Paired per-position comparisons (`scripts/paired.py`); standard M2 suite of 50 positions × 20 s (8 threads) and × 10 s (1 thread). |
