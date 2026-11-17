#!/usr/bin/env python3
"""Compare benchmark results: scripts/compare.py BEFORE AFTER (labels in bench/results).

Prints a markdown table with mean best length at checkpoints, mean final length,
mean first-solution time/length and nodes per position, for the -mt and -st suites."""
import json, sys, os

def load(label, suite):
    p = os.path.join("bench", "results", f"{label}-{suite}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def at(hist, t):
    v = [l for (s, l) in hist if s <= t]
    return min(v) if v else None

def summary(d):
    pos = d["positions"]
    n = len(pos)
    out = {}
    for c in [0.25, 0.5, 1, 2, 5, 10, 12]:
        if c > d["time"] + 1e-9:
            continue
        vals = [at(p["history"], c) for p in pos]
        out[f"length at {c} s"] = sum(vals) / n if all(v is not None for v in vals) else None
    out["final length"] = sum(p["final"] for p in pos) / n
    firsts = [p["history"][0] for p in pos if p["history"]]
    if firsts:
        out["first solution (s)"] = sum(f[0] for f in firsts) / len(firsts)
        out["first solution (moves)"] = sum(f[1] for f in firsts) / len(firsts)
    for i, ph in enumerate(["p1", "p2", "p3"]):
        out[f"nodes {ph} / position"] = sum(p["nodes"][i] for p in pos) / n
    return out

def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float) and v >= 1e5:
        return f"{v:.3e}"
    return f"{v:.3f}"

before, after = sys.argv[1], sys.argv[2]
for suite in ["mt", "st"]:
    a, b = load(before, suite), load(after, suite)
    if a is None or b is None:
        continue
    sa, sb = summary(a), summary(b)
    print(f"\n{'Multi' if suite == 'mt' else 'Single'}-threaded ({len(a['positions'])} positions, {a['time']} s, {a['threads']} threads)\n")
    print(f"| metric | {before} | {after} |")
    print("|---|---:|---:|")
    for k in sa:
        print(f"| {k} | {fmt(sa.get(k))} | {fmt(sb.get(k))} |")
