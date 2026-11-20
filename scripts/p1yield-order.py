#!/usr/bin/env python3
"""Offline phase-1 job-order evaluation from examples/p1yield records (opt/p1-order2).

    scripts/p1yield-order.py FILE... [--limit 26] [--beta 1] [--fr 0.05,0.1,0.2,0.3,0.5]
        [--file-limit 26] [--base RUN.json]

Per cube, the target is its last recorded length (p1yield with MAXN: the first whose nodes
exceed the budget, i.e. the length a run cuts short), predicted from the length before. For
each predictor (move order, the yield per 3/4/5-move prefix at the previous length with
hierarchical smoothing, signal variants, oracles), the mean share of the target length's
kept candidates (L1 + h2 <= limit, weighted beta^slack) in the first fraction f of its nodes,
and per cube, with the cut point from the run's phase-1 nodes (--base: a bench JSON of the
same positions, default bench/results/m2-L-live1_50x20.json), all kept candidates up to the
cut relative to the 3-move yield (master). --file-limit: the LIMIT p1yield wrote with.
"""
import sys
import numpy as np

DT = np.dtype([('cube', 'u1'), ('len', 'u1'), ('plen', 'u1'), ('d', 'u1'), ('code', '<u4'),
               ('nodes', '<u4'), ('eps', '<u4'), ('c', '<u2', (6,)), ('paired', 'u1'),
               ('bad', 'u1'), ('pad', '<u2')])

args = sys.argv[1:]
opt = {'--limit': '26', '--beta': '1', '--fr': '0.05,0.1,0.2,0.3,0.5', '--file-limit': '26',
       '--target': 'last', '--show': '', '--base': ''}
files = []
i = 0
while i < len(args):
    if args[i] in opt:
        opt[args[i]] = args[i + 1]
        i += 2
    else:
        files.append(args[i])
        i += 1
LIMIT = int(opt['--limit'])
FLIMIT = int(opt['--file-limit'])
BETA = float(opt['--beta'])
FRS = [float(x) for x in opt['--fr'].split(',')]

recs = np.concatenate([np.fromfile(f, dtype=DT) for f in files])


def kept(c, limit, beta=1.0):
    """Candidates with L1 + h2 <= limit, weighted by beta^slack (count columns: index k has
    L1 + h2 = FLIMIT - 3 + k, index 0 also anything less)."""
    w = np.zeros(len(c))
    for k in range(6):
        v = FLIMIT - 3 + k
        if v > limit:
            continue
        s = limit - v
        w += c[:, k] * beta ** s
    return w


def prefix(code, n):
    return (code & ((1 << (5 * n)) - 1)).astype(np.int64)


def lex(code, n=5):
    key = np.zeros(len(code), dtype=np.int64)
    for j in range(n):
        key = key * 32 + ((code.astype(np.int64) >> (5 * j)) & 31)
    return key


def sums_by(keys, vals):
    u, inv = np.unique(keys, return_inverse=True)
    s = np.zeros(len(u))
    np.add.at(s, inv, vals)
    return u, s


def lookup(u, s, keys):
    j = np.searchsorted(u, keys)
    j = np.minimum(j, len(u) - 1)
    hit = u[j] == keys
    return np.where(hit, s[j], 0.0)


def smooth(prev_keys, sig, nodes, cur_keys, prior, m):
    """(sum sig + prior * m) / (sum nodes + m) per current key (sums over prev records)."""
    u, ks = sums_by(prev_keys, sig)
    _, ns = sums_by(prev_keys, nodes)
    return (lookup(u, ks, cur_keys) + prior * m) / (lookup(u, ns, cur_keys) + m)


def mean_nodes(keys, nodes):
    return sums_by(keys, nodes)[1].mean()


def hier(prv, sig, cur, levels=(3,), mult=1.0, base=None):
    """Hierarchical smoothed rate of `sig` per node: 3-prefix toward the global rate, then
    each finer prefix toward its parent's estimate."""
    pn = prv['nodes'].astype(float)
    rate = sig.sum() / max(pn.sum(), 1)
    est = np.full(len(cur), rate) if base is None else base
    sub = prv['plen'] == 5
    for g in levels:
        pk, ck = prefix(prv['code'], g), prefix(cur['code'], g)
        m = mean_nodes(pk[sub], pn[sub]) * mult
        est = smooth(pk, sig, pn, ck, est, m)
    return est


FCUT = [None]


def capture(score, cur, w):
    base = np.argsort(lex(cur['code']), kind='stable')
    score, nodes, w = score[base], cur['nodes'][base].astype(float), w[base]
    idx = np.argsort(-score, kind='stable')
    cn = np.cumsum(nodes[idx]) / nodes.sum()
    cw = np.cumsum(w[idx]) / max(w.sum(), 1e-300)
    fs = FRS + ([FCUT[0]] if FCUT[0] is not None else [])
    return np.array([cw[min(np.searchsorted(cn, f), len(cw) - 1)] if f > 0 else 0.0 for f in fs])


import json
BASE = json.load(open(opt.get('--base') or 'bench/results/m2-L-live1_50x20.json'))
FRAC = 8
rows = {}
info = []
tot = {}
for cube in np.unique(recs['cube']):
    rc = recs[recs['cube'] == cube]
    lens = sorted(np.unique(rc['len']).tolist())
    L = lens[-1] if opt['--target'] == 'last' else lens[-1] + int(opt['--target'])
    if L - 1 not in lens:
        continue
    cur, prv = rc[rc['len'] == L], rc[rc['len'] == L - 1]
    pp = rc[rc['len'] == L - 2] if L - 2 in lens else None
    wc = kept(cur['c'].astype(float), LIMIT, BETA)
    if wc.sum() == 0:
        continue
    before = sum(rc[rc['len'] == l]['nodes'].astype(float).sum() for l in lens if l < L) * FRAC
    size = cur['nodes'].astype(float).sum() * FRAC
    ntot = BASE['positions'][int(cube)]['nodes'][0]
    f = (ntot - before) / size
    if f > 1:
        print(f"cube {cube}: cut level past {L} (f {f:.2f}), skipped")
        continue
    FCUT[0] = max(f, 0.0)
    wbefore = sum(kept(rc[rc['len'] == l]['c'].astype(float), LIMIT, BETA).sum() for l in lens if l < L)
    info.append((int(cube), L, int(cur['nodes'].astype(float).sum()), wc.sum(),
                 kept(cur['c'].astype(float), LIMIT).sum(),
                 kept(prv['c'].astype(float), LIMIT).sum()))
    pc = prv['c'].astype(float)
    pn = prv['nodes'].astype(float)
    res = {}
    res['move'] = capture(-lex(cur['code']).astype(float), cur, wc)
    sig_k = kept(pc, LIMIT)          # master: kept at L-1
    y3 = hier(prv, sig_k, cur, (3,))
    res['yield3'] = capture(y3, cur, wc)
    res['yield4'] = capture(hier(prv, sig_k, cur, (3, 4)), cur, wc)
    res['yield5'] = capture(hier(prv, sig_k, cur, (3, 4, 5)), cur, wc)
    for dl in (-1, 1, 2):
        s = kept(pc, LIMIT + dl)
        res[f'y3 lim{dl:+d}'] = capture(hier(prv, s, cur, (3,)), cur, wc)
        res[f'y5 lim{dl:+d}'] = capture(hier(prv, s, cur, (3, 4, 5)), cur, wc)
    sig_w = kept(pc, LIMIT, 4.0)
    res['y3 w4'] = capture(hier(prv, sig_w, cur, (3,)), cur, wc)
    res['y5 w4'] = capture(hier(prv, sig_w, cur, (3, 4, 5)), cur, wc)
    # Endpoint rate x kept share (both smoothed, 3- to 5-prefix).
    eps = prv['eps'].astype(float)
    er = hier(prv, eps, cur, (3, 4, 5))
    for dl in (0, 1):
        kk = kept(pc, LIMIT + dl)
        # kept per endpoint, hierarchical over endpoints as the "nodes".
        prv2 = prv.copy()
        prv2['nodes'] = np.minimum(eps, 2**32 - 1).astype(np.uint32)
        ks = hier(prv2, kk, cur, (3, 4, 5))
        res[f'eps x share{dl:+d}'] = capture(er * ks, cur, wc)
    if pp is not None:
        # L-1 and L-2 together (L-2 scaled to L-1's overall rate).
        s2 = kept(pp['c'].astype(float), LIMIT)
        sc = sig_k.sum() / max(pn.sum(), 1) / max(s2.sum() / max(pp['nodes'].astype(float).sum(), 1), 1e-300)
        both = np.concatenate([prv, pp])
        sig_b = np.concatenate([sig_k, s2 * sc])
        res['y3 +L-2'] = capture(hier(both, sig_b, cur, (3,)), cur, wc)
        res['y5 +L-2'] = capture(hier(both, sig_b, cur, (3, 4, 5)), cur, wc)
    for mult in (0.25, 4.0):
        res[f'y5 m{mult:g}'] = capture(hier(prv, sig_k, cur, (3, 4, 5), mult), cur, wc)
    # Oracles.
    for g in (3, 4, 5):
        ck = prefix(cur['code'], g)
        u, ws = sums_by(ck, wc)
        _, ns = sums_by(ck, cur['nodes'].astype(float))
        res[f'oracle{g}'] = capture(lookup(u, ws, ck) / np.maximum(lookup(u, ns, ck), 1), cur, wc)
    rows[int(cube)] = {k: v[:len(FRS)] for k, v in res.items()}
    tot[int(cube)] = (FCUT[0], {k: wbefore + v[-1] * wc.sum() for k, v in res.items()})

print("cube L nodes(sampled) weighted-kept kept(L) kept(L-1)")
for x in info:
    print(*x)
names = list(next(iter(rows.values())).keys())
y3 = np.array([r['yield3'] for r in rows.values()])
print(f"\n{len(rows)} cubes, limit {LIMIT}, beta {BETA}: mean share of the target length's (weighted) kept candidates in the first f of its nodes; (diff vs yield3 +- s.e.)")
print(f"{'order':14s} " + " ".join(f"{'f=' + format(f, 'g'):>17s}" for f in FRS))
for nm in names:
    arr = np.array([r[nm] for r in rows.values()])
    d = arr - y3
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else d[0] * 0
    print(f"{nm:14s} " + " ".join(f"{a:.3f} ({b:+.3f}+-{s:.3f})" for a, b, s in zip(arr.mean(axis=0), d.mean(axis=0), se)))

print("\nper cube: f at the cut, then total (weighted) kept up to the cut relative to yield3")
names2 = list(next(iter(tot.values()))[1].keys())
print("cube f " + " ".join(f"{n[:9]:>9s}" for n in names2))
rel = {n: [] for n in names2}
for c, (f, t) in tot.items():
    print(f"{c:4d} {f:.2f} " + " ".join(f"{t[n] / t['yield3']:9.3f}" for n in names2))
    for n in names2:
        rel[n].append(t[n] / t['yield3'])
print("mean      " + " ".join(f"{np.mean(rel[n]):9.3f}" for n in names2))
print("geomean   " + " ".join(f"{np.exp(np.mean(np.log(np.maximum(rel[n], 1e-9)))):9.3f}" for n in names2))
