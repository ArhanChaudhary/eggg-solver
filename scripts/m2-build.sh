#!/bin/bash
# Build git REF on the M2 laptop as ~/eggg/bin/eggg-LABEL (untimed: nice 19, no lock).
#   scripts/m2-build.sh LABEL REF
set -e
label=${1:?usage: m2-build.sh LABEL REF}; ref=${2:?}
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed22519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
$M2 'true' >/dev/null 2>&1
tmp=$(mktemp -d)
git archive "$ref" | tar x -C "$tmp"
rsync -az --delete -e "$E" "$tmp/" arhan@192.168.191.121:eggg/src/
rm -rf "$tmp"
$M2 ". ~/fto/env.sh; export CARGO_TARGET_DIR=~/eggg/target TMPDIR=~/eggg/tmp; mkdir -p ~/eggg/tmp ~/eggg/bin;
     find ~/eggg/src -name "*.rs" -exec touch {} +; cd ~/eggg/src && nice -n 19 cargo build --release 2>&1 | tail -2 && cp ~/eggg/target/release/eggg-solver ~/eggg/bin/eggg-$label && echo built eggg-$label" 2>&1 | grep -E "built|error|warning|Finished"
