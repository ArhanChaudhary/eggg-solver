#!/bin/bash
# Paired runs on this machine inside one exclusive /tmp/cube-bench.lock unit; JSON in
# bench/results/x86-L-NAME.json for scripts/paired.py. Run from the repository root.
#   scripts/x86-long.sh N SECONDS SEED NAME[=BUILD][:flags+separated+by+plus]...
# (EGGG_BENCH_TIER sets the tier unless the flags give one; default huge.)
set -e
n=${1:?N}; secs=${2:?SECONDS}; seed=${3:?SEED}; shift 3
tier=${EGGG_BENCH_TIER:-huge}
mkdir -p bench/results
cmds=""
for spec in "$@"; do
  ns=${spec%%:*}; flags=""; [[ $spec == *:* ]] && flags=${spec#*:}; flags=${flags//+/ }
  name=${ns%%=*}; build=$name; [[ $ns == *=* ]] && build=${ns#*=}
  tflag="--tier $tier"; [[ $flags == *--tier* ]] && tflag=""
  cmds="$cmds echo == $name; /data3/eggg-bin/eggg-$build bench -N $n -t $secs --seed $seed $tflag $flags --tables /data3/eggg-tables --json bench/results/x86-L-$name.json 2>&1 | grep -E 'tables:|final|split|mean length';"
done
flock -x /tmp/cube-bench.lock bash -c "$cmds"
