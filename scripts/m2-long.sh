#!/bin/bash
# Paired runs on the M2 inside one exclusive /tmp/m2-bench.lock unit (MemAvailable checked
# inside the lock, so a queued run waits instead of giving up); JSON copied back to
# bench/results/m2-L-NAME.json for scripts/paired.py. Run from the repository root.
#   scripts/m2-long.sh N SECONDS SEED NAME[=BUILD][:flags+separated+by+plus]...
set -e
n=${1:?N}; secs=${2:?SECONDS}; seed=${3:?SEED}; shift 3
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
cmds="a=\$(grep MemAvailable /proc/meminfo | awk '{print int(\$2/1024/1024)}'); echo M2 MemAvailable \$a GB; [ \$a -ge 10 ] || { echo not starting; exit 1; }; mkdir -p ~/eggg/results; cd ~/eggg;"
names=()
for spec in "$@"; do
  ns=${spec%%:*}; flags=""; [[ $spec == *:* ]] && flags=${spec#*:}; flags=${flags//+/ }
  name=${ns%%=*}; build=$name; [[ $ns == *=* ]] && build=${ns#*=}
  names+=("$name")
  cmds="$cmds echo == $name; ./bin/eggg-$build bench -N $n -t $secs --seed $seed $flags --tables ~/eggg/tables --json results/L-$name.json 2>&1 | grep -E 'final|split|mean length';"
done
tmp=$(mktemp)
printf '%s\n' "$cmds" > "$tmp"
$M2 'mkdir -p ~/eggg/tmp' >/dev/null
rsync -az -e "$E" "$tmp" arhan@192.168.191.121:eggg/tmp/long-$$.sh
rm -f "$tmp"
$M2 "flock -x /tmp/m2-bench.lock bash ~/eggg/tmp/long-$$.sh; rm -f ~/eggg/tmp/long-$$.sh" 2>&1 | grep -v "^\s*$"
mkdir -p bench/results
for name in "${names[@]}"; do
  rsync -az -e "$E" arhan@192.168.191.121:eggg/results/L-$name.json bench/results/m2-L-$name.json
done
