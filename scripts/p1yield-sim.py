#!/usr/bin/env python3
"""Simulate phase-1 schedules up to each cube's node budget (the phase-1 nodes of a bench run
of the same positions, --base): level-complete (each length best first by its predictor,
GRAN-move prefixes) vs best first across two lengths (the cut length's head before the
previous length's tail; calibrated with the true totals), from examples/p1yield records.

    scripts/p1yield-sim.py FILE... [--limit 26] [--beta 1] [--gran 4] [--base RUN.json]
Levels per cube: C = the last recorded, B = C - 1, A = C - 2 (A predicts B, B predicts C).
"""
import sys, json
import numpy as np

DT = np.dtype([('cube', 'u1'), ('len', 'u1'), ('plen', 'u1'), ('d', 'u1'), ('code', '<u4'),
               ('nodes', '<u4'), ('eps', '<u4'), ('c', '<u2', (6,)), ('paired', 'u1'),
               ('bad', 'u1'), ('pad', '<u2')])
args = sys.argv[1:]
opt = {'--limit': '26', '--beta': '1', '--gran': '4', '--file-limit': '26', '--frac': '8',
       '--base': 'bench/results/m2-L-live1_50x20.json', '--scale': '1'}
files = []
i = 0
while i < len(args):
    if args[i] in opt:
        opt[args[i]] = args[i + 1]; i += 2
    else:
        files.append(args[i]); i += 1
LIMIT, FLIMIT, BETA, GRAN = int(opt['--limit']), int(opt['--file-limit']), float(opt['--beta']), int(opt['--gran'])
FRAC = float(opt['--frac'])
SCALE = float(opt['--scale'])  # budget multiplier (time scaling)
BASE = json.load(open(opt['--base']))
recs = np.concatenate([np.fromfile(f, dtype=DT) for f in files])


def kept(c, limit, beta=1.0):
    w = np.zeros(len(c))
    for k in range(6):
        v = FLIMIT - 3 + k
        if v <= limit:
            w += c[:, k] * beta ** (limit - v)
    return w


def prefix(code, n):
    return (code.astype(np.int64) & ((1 << (5 * n)) - 1))


def sums_by(keys, vals):
    u, inv = np.unique(keys, return_inverse=True)
    s = np.zeros(len(u)); np.add.at(s, inv, vals)
    return u, s


def lookup(u, s, keys):
    j = np.minimum(np.searchsorted(u, keys), len(u) - 1)
    return np.where(u[j] == keys, s[j], 0.0)


def units(r, w):
    """Aggregate records to GRAN-prefix units: keys, nodes, weight (sorted by key)."""
    sub = r['plen'] == 5
    k = prefix(r['code'][sub], GRAN)
    u, n = sums_by(k, r['nodes'][sub].astype(float))
    _, ww = sums_by(k, w[sub])
    return u, n, ww


def predict(prev, wprev, ukeys):
    """Hierarchical smoothed kept/node of `ukeys` (GRAN-prefix units) from the previous level."""
    pn = prev['nodes'].astype(float)
    rate = wprev.sum() / max(pn.sum(), 1)
    est = np.full(len(ukeys), rate)
    sub = prev['plen'] == 5
    for g in range(3, GRAN + 1):
        pk = prefix(prev['code'], g)
        u, ks = sums_by(pk, wprev); _, ns = sums_by(pk, pn)
        _, nsub = sums_by(pk[sub], pn[sub])
        m = nsub.mean()
        ck = ukeys & ((1 << (5 * g)) - 1)
        est = (lookup(u, ks, ck) + est * m) / (lookup(u, ns, ck) + m)
    return est


def lexkey(keys, n):
    out = np.zeros(len(keys), dtype=np.int64)
    for j in range(n):
        out = out * 32 + ((keys >> (5 * j)) & 31)
    return out


def take(order_nodes, order_w, budget):
    """Weight collected processing units in order until `budget` nodes (partial last unit
    pro rata)."""
    cn = np.cumsum(order_nodes)
    j = np.searchsorted(cn, budget)
    got = order_w[:j].sum()
    if j < len(order_nodes):
        prev = cn[j - 1] if j > 0 else 0.0
        got += order_w[j] * (budget - prev) / max(order_nodes[j], 1)
    return got


out = []
for cube in np.unique(recs['cube']):
    rc = recs[recs['cube'] == cube]
    lens = sorted(np.unique(rc['len']).tolist())
    C = lens[-1]
    B, A = C - 1, C - 2
    if A not in lens:
        continue
    ra, rb, rcc = (rc[rc['len'] == l] for l in (A, B, C))
    wa, wb, wcc = (kept(r['c'].astype(float), LIMIT, BETA) for r in (ra, rb, rcc))
    before = sum(rc[rc['len'] == l]['nodes'].astype(float).sum() for l in lens if l < B) * FRAC
    budget = (BASE['positions'][int(cube)]['nodes'][0] * SCALE - before) / FRAC
    if budget <= 0:
        continue
    ub, nb, kb = units(rb, wb)
    uc, nc, kc = units(rcc, wcc)
    if budget > nb.sum() + nc.sum():
        print(f"cube {cube}: budget past level {C}, skipped")
        continue
    pb = predict(ra, wa, ub)
    pc = predict(rb, wb, uc)
    # Calibration (cheat: the true totals): expected yield of a unit = score * k.
    kb_cal = kb.sum() / max((pb * nb).sum(), 1e-300)
    kc_cal = kc.sum() / max((pc * nc).sum(), 1e-300)
    res = {}
    lb, lc = lexkey(ub, GRAN), lexkey(uc, GRAN)
    # Level-complete, move order.
    ob, oc = np.argsort(lb, kind='stable'), np.argsort(lc, kind='stable')
    res['move'] = take(np.concatenate([nb[ob], nc[oc]]), np.concatenate([kb[ob], kc[oc]]), budget)
    # Level-complete, best first per level.
    ob = np.lexsort((lb, -pb)); oc = np.lexsort((lc, -pc))
    res['level'] = take(np.concatenate([nb[ob], nc[oc]]), np.concatenate([kb[ob], kc[oc]]), budget)
    # Best first across both levels (calibrated scores).
    s = np.concatenate([pb * kb_cal, pc * kc_cal])
    nn = np.concatenate([nb, nc]); ww = np.concatenate([kb, kc])
    lev = np.concatenate([np.zeros(len(nb)), np.ones(len(nc))])
    o = np.lexsort((np.concatenate([lb, lc]), lev, -s))
    res['cross'] = take(nn[o], ww[o], budget)
    # Cross with a damped next level (its scores x 0.5): robustness to calibration.
    for damp in (0.5, 2.0):
        s2 = np.concatenate([pb * kb_cal, pc * kc_cal * damp])
        o = np.lexsort((np.concatenate([lb, lc]), lev, -s2))
        res[f'cross x{damp:g}'] = take(nn[o], ww[o], budget)
    # Oracle cross (true unit yields).
    s3 = ww / np.maximum(nn, 1)
    o = np.argsort(-s3, kind='stable')
    res['oracle'] = take(nn[o], ww[o], budget)
    f = (budget - nb.sum()) / nc.sum()
    out.append((int(cube), B, C, f, res))

names = list(out[0][4].keys())
print(f"limit {LIMIT} beta {BETA} gran {GRAN} scale {SCALE}: (weighted) kept collected within the budget, relative to 'level'")
print("cube  B  C  f(C) " + " ".join(f"{n:>10s}" for n in names))
rel = {n: [] for n in names}
for cube, B, C, f, res in out:
    print(f"{cube:4d} {B:2d} {C:2d} {f:5.2f} " + " ".join(f"{res[n] / res['level']:10.3f}" for n in names))
    for n in names:
        rel[n].append(res[n] / res['level'])
print("mean            " + " ".join(f"{np.mean(rel[n]):10.3f}" for n in names))
print("geomean         " + " ".join(f"{np.exp(np.mean(np.log(rel[n]))):10.3f}" for n in names))
