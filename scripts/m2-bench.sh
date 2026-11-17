#!/bin/bash
# Run the standard benchmark suites on the M2 for several builds, interleaved, inside one
# exclusive /tmp/m2-bench.lock unit; copy the JSON back to bench/results/m2-NAME-{mt,st}.json.
#   scripts/m2-bench.sh NAME[=BUILD][:FLAGS]...
# e.g. scripts/m2-bench.sh p1 joint:--p2-joint   (BUILD defaults to NAME; FLAGS are extra
# solver flags, commas for spaces).
set -e
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed22519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
h48=${EGGG_BENCH_H48:-6}
cmds="mkdir -p ~/eggg/results; cd ~/eggg; awk '/MemAvailable/' /proc/meminfo;"
names=()
for suite in mt st; do
  for spec in "$@"; do
    ns=${spec%%:*}; flags=""; [[ $spec == *:* ]] && flags=${spec#*:}; flags=${flags//,/ }
    name=${ns%%=*}; build=$name; [[ $ns == *=* ]] && build=${ns#*=}
    [ $suite = mt ] && names+=("$name")
    if [ $suite = mt ]; then args="-N 10 -t 5 --seed 2026"; else args="-N 4 -t 12 -j 1 --seed 2027"; fi
    hflag="--h48 $h48"; [[ $flags == *--h48* ]] && hflag=""
    cmds="$cmds echo == $name $suite; ./bin/eggg-$build bench $args $hflag $flags --tables ~/eggg/tables --json results/$name-$suite.json 2>&1 | grep -v 'bfs depth';"
  done
done
$M2 "flock -x /tmp/m2-bench.lock bash -c \"$cmds\"" 2>&1 | grep -v "^\s*$" | grep -E "==|mean|first|final|nodes|positions|Mem|error"
mkdir -p bench/results
for name in "${names[@]}"; do
  for suite in mt st; do
    rsync -az -e "$E" arhan@192.168.191.121:eggg/results/$name-$suite.json bench/results/m2-$name-$suite.json
  done
done
