#!/bin/bash
# Sync this worktree to ~/eggg/src-p1h2 on the M2 and build the examples p1h2 and p1check and
# the solver there (under the lock, own target dir); the solver is copied to
# ~/eggg/bin/eggg-LABEL.  usage: m2-example.sh LABEL
set -e
label=${1:?}
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed22519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
$M2 'true' >/dev/null 2>&1
cd /data3/eggg-wt-p1h2
git rev-parse HEAD > BUILD_REF; git diff --stat >> BUILD_REF
rsync -az --delete --exclude target --exclude tables -e "$E" ./ arhan@192.168.191.121:eggg/src-p1h2/
$M2 ". ~/fto/env.sh; export CARGO_TARGET_DIR=~/eggg/target-p1h2 TMPDIR=~/eggg/tmp; mkdir -p ~/eggg/tmp;
     cd ~/eggg/src-p1h2 && flock -x /tmp/m2-bench.lock bash -c 'set -o pipefail; touch src/*.rs examples/*.rs; nice -n 19 cargo build --release --example p1h2 --example p1check --bin eggg-solver 2>&1 | tee ~/eggg/tmp/build-p1h2.log | tail -2 && grep -q \"Compiling eggg-solver\" ~/eggg/tmp/build-p1h2.log && cp ~/eggg/target-p1h2/release/eggg-solver ~/eggg/bin/eggg-$label && cp BUILD_REF ~/eggg/bin/eggg-$label.ref && echo built eggg-$label'"
