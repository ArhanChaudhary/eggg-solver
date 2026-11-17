#!/bin/bash
# Standard suites on this machine for several builds, interleaved, inside one exclusive
# /tmp/cube-bench.lock unit: bench/results/x86-NAME-{mt,st}.json.
#   scripts/x86-bench.sh NAME[=BUILD][:FLAGS]...   (FLAGS: extra solver flags, commas for spaces)
set -e
mkdir -p bench/results
tier=${EGGG_BENCH_TIER:-huge}
run() {
  for suite in mt st; do
    for spec in "$@"; do
      ns=${spec%%:*}; flags=""; [[ $spec == *:* ]] && flags=${spec#*:}; flags=${flags//,/ }
      name=${ns%%=*}; build=$name; [[ $ns == *=* ]] && build=${ns#*=}
      hflag="--tier $tier"; [[ $flags == *--tier* ]] && hflag=""
      if [ $suite = mt ]; then args="-N 10 -t 5 --seed 2026"; else args="-N 4 -t 12 -j 1 --seed 2027"; fi
      echo "== $name $suite"
      /data3/eggg-bin/eggg-$build bench $args $hflag $flags --tables /data3/eggg-tables --json bench/results/x86-$name-$suite.json 2>&1 | grep -E "positions|mean length at  (1|2|5)|first|final|nodes|split"
    done
  done
}
export -f run
export tier
flock -x /tmp/cube-bench.lock bash -c "$(declare -f run); run $*"
