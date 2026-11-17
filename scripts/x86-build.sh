#!/bin/bash
# Build git REF as /data3/eggg-bin/eggg-LABEL (shared bench lock, nice).
#   scripts/x86-build.sh LABEL REF
set -e
label=${1:?usage: x86-build.sh LABEL REF}; ref=${2:?}
dir=/data3/eggg-build/$label
rm -rf "$dir"; mkdir -p "$dir"
git archive "$ref" | tar x -C "$dir"
find "$dir" -name "*.rs" -exec touch {} +
cp .cargo/config.toml "$dir/.cargo/config.toml" 2>/dev/null || true
(cd "$dir" && CARGO_TARGET_DIR=/data3/eggg-build/target flock -s /tmp/cube-bench.lock nice cargo build --release 2>&1 | tail -1)
cp /data3/eggg-build/target/release/eggg-solver /data3/eggg-bin/eggg-$label
echo built eggg-$label
