#!/bin/bash
# One locked M2 session (method/p1-h2): sync, build the examples and the solver, then run the
# steps in STEPS (a bash snippet run in ~/eggg/src-p1h2 with $T = the target dir), all under
# one hold of /tmp/m2-bench.lock (keep it under ~15 min).  usage: m2-session.sh NAME STEPS
set -e
name=${1:?}; steps=${2:?}
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed22519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
$M2 'true' >/dev/null 2>&1
cd /data3/eggg-wt-p1h2
printf '%s\n' "$steps" > steps-$name.sh
rsync -az --delete --exclude target --exclude tables -e "$E" ./ arhan@192.168.191.121:eggg/src-p1h2/
$M2 "export BUILD=${BUILD:-1}; . ~/fto/env.sh; export CARGO_TARGET_DIR=~/eggg/target-p1h2 TMPDIR=~/eggg/tmp; mkdir -p ~/eggg/tmp ~/eggg/scratch-p1h2;
     cd ~/eggg/src-p1h2 && ln -sfn ~/eggg/tables tables && flock -x /tmp/m2-bench.lock bash -c 'set -o pipefail; date; if [ ${BUILD:-1} = 1 ]; then touch src/*.rs examples/*.rs; nice -n 19 cargo build --release --example p1h2 --example p1check --bin eggg-solver 2>&1 | tail -1 || exit 1; fi; T=~/eggg/target-p1h2/release; . ./steps-$name.sh; date' > ~/eggg/scratch-p1h2/$name.log 2>&1; cat ~/eggg/scratch-p1h2/$name.log"
