#!/usr/bin/env python3
"""Paired comparison of two benchmark runs on the same positions (same seed and count).

    scripts/paired.py bench/results/A.json bench/results/B.json [checkpoints...]

For each checkpoint: mean best length in A and B over the positions solved in both, the
mean difference B - A, and how many positions B made shorter / longer.
"""
import json, sys

def best_at(pos, t):
    b = None
    for when, length in pos["history"]:
        if when <= t:
            b = length
    return b

a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert a["seed"] == b["seed"] and len(a["positions"]) == len(b["positions"]), "not paired"
cps = [float(x) for x in sys.argv[3:]] or [0.5, 1, 2, 5, a["time"]]
print(f"| t (s) | n | {sys.argv[1].split('/')[-1]} | {sys.argv[2].split('/')[-1]} | B - A | B shorter | B longer |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for t in sorted(set(c for c in cps if c <= a["time"])):
    pairs = [(best_at(p, t), best_at(q, t)) for p, q in zip(a["positions"], b["positions"])]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if not pairs:
        continue
    n = len(pairs)
    ma = sum(x for x, _ in pairs) / n
    mb = sum(y for _, y in pairs) / n
    print(f"| {t:g} | {n} | {ma:.3f} | {mb:.3f} | {mb - ma:+.3f} | {sum(y < x for x, y in pairs)} | {sum(y > x for x, y in pairs)} |")
