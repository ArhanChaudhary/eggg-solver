#!/bin/bash
# Run the standard benchmark suites on the M2 for several builds, interleaved, inside one
# exclusive /tmp/m2-bench.lock unit; copy the JSON back to bench/results/m2-LABEL-{mt,st}.json.
#   scripts/m2-bench.sh LABEL...
set -e
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed22519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
h48=${EGGG_BENCH_H48:-6}
cmds="mkdir -p ~/eggg/results; cd ~/eggg; awk '/MemAvailable/' /proc/meminfo;"
for suite in mt st; do
  for l in "$@"; do
    if [ $suite = mt ]; then args="-N 10 -t 5 --seed 2026"; else args="-N 4 -t 12 -j 1 --seed 2027"; fi
    cmds="$cmds echo == $l $suite; ./bin/eggg-$l bench $args --h48 $h48 --tables ~/eggg/tables --json results/$l-$suite.json 2>&1 | grep -v 'bfs depth';"
  done
done
$M2 "flock -x /tmp/m2-bench.lock bash -c \"$cmds\"" 2>&1 | grep -v "^\s*$" | grep -E "==|mean|first|final|nodes|positions|Mem|error"
mkdir -p bench/results
for l in "$@"; do
  for suite in mt st; do
    rsync -az -e "$E" arhan@192.168.191.121:eggg/results/$l-$suite.json bench/results/m2-$l-$suite.json
  done
done
