#!/bin/bash
# Profile-guided (LLVM PGO) release build of git REF: an instrumented build, a short training
# run on the target machine (bench -N 3 -t 10, all threads, its default tier), the profile
# merged by llvm-profdata, and the release build with it (x86: tier huge, as x86-bench.sh).
#   scripts/pgo-build.sh x86 LABEL REF   built and trained here: /data3/eggg-bin/eggg-LABEL
#   scripts/pgo-build.sh m2 LABEL REF    cross-built here for aarch64 (target-cpu=apple-m2, the
#                                        features of the M2's native builds), trained on the
#                                        M2 under its exclusive lock: ~/eggg/bin/eggg-LABEL
# Needs this machine's Rust toolchain with the llvm-tools component (llvm-profdata, same LLVM
# as rustc) and, for m2, the aarch64-unknown-linux-gnu target and aarch64-linux-gnu-gcc. A
# profile belongs to one source tree and set of flags: every commit needs its own training run
# (one minute on the machine it is for; a profile from another machine's run also works, as
# the instrumented code is the same up to the target, but is trained on that machine's mix).
set -e
mode=${1:?usage: pgo-build.sh x86|m2 LABEL REF}; label=${2:?}; ref=${3:?}
profdata=$(ls "$(rustc --print sysroot)"/lib/rustlib/*/bin/llvm-profdata | head -1)
[ -x "$profdata" ] || { echo "llvm-profdata missing: rustup component add llvm-tools"; exit 1; }
work=/data3/eggg-build/pgo-$label
rm -rf "$work"; mkdir -p "$work/src" "$work/raw"
git archive "$ref" | tar x -C "$work/src"
git rev-parse "$ref" > "$work/BUILD_REF"
find "$work/src" -name "*.rs" -exec touch {} +
export CARGO_TARGET_DIR=$work/target
train="bench -N 3 -t 10 --seed 4242"
build() { # RUSTFLAGS for the profile step; the target's own flags are added
  if [ "$mode" = m2 ]; then
    (cd "$work/src" && CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
      CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-cpu=apple-m2 $1" \
      nice -n 19 cargo build --release -j 6 --target aarch64-unknown-linux-gnu 2>&1 | tail -1)
    bin=$CARGO_TARGET_DIR/aarch64-unknown-linux-gnu/release/eggg-solver
  else
    # (RUSTFLAGS would replace the .cargo/config.toml target flags: repeat them.)
    (cd "$work/src" && RUSTFLAGS="-C target-cpu=x86-64-v3 $1" flock -s /tmp/cube-bench.lock \
      nice -n 19 cargo build --release 2>&1 | tail -1)
    bin=$CARGO_TARGET_DIR/release/eggg-solver
  fi
}
build "-Cprofile-generate=$work/raw"
if [ "$mode" = m2 ]; then
  M2=/data3/eggg-tools/m2
  E="ssh -i /home/super/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
  H=arhan@192.168.191.121
  $M2 "rm -rf ~/eggg/pgo-$label; mkdir -p ~/eggg/pgo-$label/raw" >/dev/null
  rsync -az -e "$E" "$bin" $H:eggg/pgo-$label/eggg-instr
  $M2 "cd ~/eggg && flock -x /tmp/m2-bench.lock bash -c 'LLVM_PROFILE_FILE=\$HOME/eggg/pgo-$label/raw/%p-%m.profraw ./pgo-$label/eggg-instr $train --tables ~/eggg/tables | grep \"mean length\" | tail -1'"
  rsync -az -e "$E" $H:eggg/pgo-$label/raw/ "$work/raw/"
else
  flock -x /tmp/cube-bench.lock "$bin" $train --tier huge --tables /data3/eggg-tables | grep "mean length" | tail -1
fi
"$profdata" merge -o "$work/merged.profdata" "$work"/raw/*.profraw
build "-Cprofile-use=$work/merged.profdata"
if [ "$mode" = m2 ]; then
  rsync -az -e "$E" "$bin" $H:eggg/bin/eggg-$label
  rsync -az -e "$E" "$work/BUILD_REF" $H:eggg/bin/eggg-$label.ref
else
  cp "$bin" /data3/eggg-bin/eggg-$label
fi
echo "built eggg-$label (PGO) from $(cat "$work/BUILD_REF")"
