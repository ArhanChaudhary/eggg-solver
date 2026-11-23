#!/bin/bash
# Build git REF as /data3/eggg-bin/eggg-LABEL (shared bench lock, nice). The package is
# cleaned first: the target directory is shared between labels and cargo's fingerprint does
# not include the source directory, so it could skip the build and copy a stale binary.
# Nothing is copied unless cargo compiled eggg-solver and the build succeeded.
#   scripts/x86-build.sh LABEL REF
set -eo pipefail
label=${1:?usage: x86-build.sh LABEL REF}; ref=${2:?}
dir=/data3/eggg-build/$label
rm -rf "$dir"; mkdir -p "$dir"
git archive "$ref" | tar x -C "$dir"
git rev-parse "$ref" > "$dir/BUILD_REF"
find "$dir" -name "*.rs" -exec touch {} +
cp .cargo/config.toml "$dir/.cargo/config.toml" 2>/dev/null || true
log=/data3/eggg-build/build-$label.log
(cd "$dir" && export CARGO_TARGET_DIR=/data3/eggg-build/target && flock -s /tmp/cube-bench.lock bash -c 'cargo clean --release -p eggg-solver -q && nice cargo build --release' > "$log" 2>&1) || { tail -5 "$log"; exit 1; }
tail -1 "$log"
grep -q "Compiling eggg-solver" "$log" || { echo "eggg-solver was not compiled; not copying"; exit 1; }
cp /data3/eggg-build/target/release/eggg-solver /data3/eggg-bin/eggg-$label
cp "$dir/BUILD_REF" /data3/eggg-bin/eggg-$label.ref
echo "built eggg-$label from $(cat "$dir/BUILD_REF")"
