#!/bin/sh
# Standard benchmark: random positions, best length after a time limit.
#   scripts/bench.sh LABEL [BINARY]
# Writes bench/results/LABEL-mt.json (all cores, 10 positions x 5 s) and
# bench/results/LABEL-st.json (1 thread, 4 positions x 12 s), and prints the summaries.
# Holds /tmp/eggg-bench.lock so that two benchmarks never overlap.
set -e
label=${1:?usage: scripts/bench.sh LABEL [BINARY]}
bin=${2:-target/release/eggg-solver}
h48=${EGGG_BENCH_H48:-6}
mkdir -p bench/results
exec 9>/tmp/eggg-bench.lock
flock 9
echo "== $label: multi-threaded (10 x 5 s)"
"$bin" bench -N 10 -t 5 --seed 2026 --h48 "$h48" --json "bench/results/$label-mt.json"
echo "== $label: single-threaded (4 x 12 s)"
"$bin" bench -N 4 -t 12 -j 1 --seed 2027 --h48 "$h48" --json "bench/results/$label-st.json"
