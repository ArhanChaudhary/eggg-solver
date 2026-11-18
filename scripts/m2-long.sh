#!/bin/bash
# Paired runs on the M2 inside one exclusive /tmp/m2-bench.lock unit (MemAvailable checked
# first); JSON copied back to bench/results/m2-L-NAME.json for scripts/paired.py.
#   scripts/m2-long.sh N SECONDS SEED NAME[=BUILD][:flags+separated+by+plus]...
set -e
n=${1:?N}; secs=${2:?SECONDS}; seed=${3:?SEED}; shift 3
M2=/data3/eggg-tools/m2
E="ssh -i /home/super/.ssh/id_ed22519 -o IdentitiesOnly=yes -o BatchMode=yes -S /data3/eggg-tools/m2.sock"
avail=$($M2 "grep MemAvailable /proc/meminfo" | awk '{print int($2/1024/1024)}')
[ "${avail:-0}" -ge 10 ] || { echo "M2 has only ${avail} GB available; not starting"; exit 1; }
echo "M2 MemAvailable ${avail} GB"
cmds="mkdir -p ~/eggg/results; cd ~/eggg;"
names=()
for spec in "$@"; do
  ns=${spec%%:*}; flags=""; [[ $spec == *:* ]] && flags=${spec#*:}; flags=${flags//+/ }
  name=${ns%%=*}; build=$name; [[ $ns == *=* ]] && build=${ns#*=}
  names+=("$name")
  cmds="$cmds echo == $name; ./bin/eggg-$build bench -N $n -t $secs --seed $seed $flags --tables ~/eggg/tables --json results/L-$name.json 2>&1 | grep -E 'final|split|mean length';"
done
$M2 "flock -x /tmp/m2-bench.lock bash -c \"$cmds\"" 2>&1 | grep -v "^\s*$"
mkdir -p bench/results
for name in "${names[@]}"; do
  rsync -az -e "$E" arhan@192.168.191.121:eggg/results/L-$name.json bench/results/m2-L-$name.json
done
