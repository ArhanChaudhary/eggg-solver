#!/bin/bash
# Build git REF on the M2 laptop as ~/eggg/bin/eggg-LABEL (nice 19, under the exclusive
# /tmp/m2-bench.lock: a release build and link can use over 1 GB, and the M2 is shared).
# Sources go to a per-label directory (~/eggg/src-LABEL), so a build queued on the lock
# never compiles sources copied later by another build; the binary's ref is printed.
# The package is cleaned first: the target directory is shared and cargo's fingerprint
# does not include the source directory, so it could skip the build and copy a stale
# binary. Nothing is copied unless cargo compiled eggg-solver and the build succeeded.
#   scripts/m2-build.sh LABEL REF
set -e
label=${1:?usage: m2-build.sh LABEL REF}; ref=${2:?}
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
$M2 'true' >/dev/null 2>&1
tmp=$(mktemp -d)
git archive "$ref" | tar x -C "$tmp"
git rev-parse "$ref" > "$tmp/BUILD_REF"
rsync -az --delete -e "$E" "$tmp/" arhan@192.168.191.121:eggg/src-$label/
rm -rf "$tmp"
$M2 ". ~/fto/env.sh; export CARGO_TARGET_DIR=~/eggg/target TMPDIR=~/eggg/tmp; mkdir -p ~/eggg/tmp ~/eggg/bin;
     cd ~/eggg/src-$label && flock -x /tmp/m2-bench.lock bash -c 'set -o pipefail; cargo clean --release -p eggg-solver -q && nice -n 19 cargo build --release 2>&1 | tee ~/eggg/tmp/build-$label.log | tail -2 && grep -q \"Compiling eggg-solver\" ~/eggg/tmp/build-$label.log && cp ~/eggg/target/release/eggg-solver ~/eggg/bin/eggg-$label && cp BUILD_REF ~/eggg/bin/eggg-$label.ref && echo built eggg-$label from \$(cat BUILD_REF)'" 2>&1 | grep -E "built|error|warning|Finished"
