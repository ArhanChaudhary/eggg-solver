#!/usr/bin/env python3
"""Online re-ranking within the cut length (examples/p1yield records): 4-move units in
4-move-yield order, re-sorted at checkpoints (fractions of the length's nodes) by the length's
own observed yield per 3-move prefix (score x (kept + a) / (expected + a), the expectation
calibrated on the units done).

    scripts/p1yield-online.py FILE... [--a A] [--limit 26]
"""
import os, sys, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p1yield-sim.py')).read().split("out = []")[0].replace("args = sys.argv[1:]", "args = [a for a in sys.argv[1:] if a not in ('--a',) and not a.replace('.','').isdigit() or a.endswith('.bin')]"))
A = [float(sys.argv[sys.argv.index('--a') + 1])] if '--a' in sys.argv else [0.5, 2.0, 8.0]
FS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
CHK = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]

def run_online(score0, p3, n, w, a, chk):
    """Process units best-first; at each checkpoint (fraction of total nodes) re-score the
    remaining units: score0 * (k_p + a) / (E_p + a) with E_p the calibrated expectation of
    p's processed units. Returns the cumulative (nodes, w) sequence."""
    tot = n.sum()
    done = np.zeros(len(n), bool)
    seq_n, seq_w = [], []
    score = score0.copy()
    used = 0.0
    for c in chk + [1.01]:
        rem = np.where(~done)[0]
        if len(rem) == 0:
            break
        o = rem[np.argsort(-score[rem], kind='stable')]
        cn = used + np.cumsum(n[o])
        j = np.searchsorted(cn, c * tot, side='right')
        take = o[:max(j, 1)]
        done[take] = True
        seq_n.append(n[take]); seq_w.append(w[take])
        used += n[take].sum()
        # Update.
        d = np.where(done)[0]
        pred = score0[d] * n[d]
        rho = w[d].sum() / max(pred.sum(), 1e-300)
        u, inv = np.unique(p3[d], return_inverse=True)
        kp = np.zeros(len(u)); ep = np.zeros(len(u))
        np.add.at(kp, inv, w[d]); np.add.at(ep, inv, pred * rho)
        j2 = np.minimum(np.searchsorted(u, p3), len(u) - 1)
        hit = u[j2] == p3
        mult = np.where(hit, (kp[j2] + a) / (ep[j2] + a), 1.0)
        score = score0 * mult
    return np.concatenate(seq_n), np.concatenate(seq_w)

def capture_seq(sn, sw, fs):
    cn = np.cumsum(sn) / sn.sum(); cw = np.cumsum(sw) / max(sw.sum(), 1e-300)
    return np.array([cw[min(np.searchsorted(cn, f), len(cw) - 1)] for f in fs])

rows = []
for cube in np.unique(recs['cube']):
    rc = recs[recs['cube'] == cube]
    lens = sorted(np.unique(rc['len']).tolist())
    C = lens[-1]
    if C - 1 not in lens:
        continue
    rb, rcc = rc[rc['len'] == C - 1], rc[rc['len'] == C]
    wb, wcc = (kept(r['c'].astype(float), LIMIT, BETA) for r in (rb, rcc))
    uc, nc, kc = units(rcc, wcc)
    o0 = np.argsort(lexkey(uc, GRAN), kind='stable')
    uc, nc, kc = uc[o0], nc[o0], kc[o0]
    pc = predict(rb, wb, uc)
    p3 = uc & ((1 << 15) - 1)
    res = {}
    o = np.argsort(-pc, kind='stable')
    res['yield4'] = capture_seq(nc[o], kc[o], FS)
    for a in A:
        sn, sw = run_online(pc, p3, nc, kc, a, CHK)
        res[f'online a{a:g}'] = capture_seq(sn, sw, FS)
    # oracle3 x yield4: score = yield4 * (true 3-prefix rate / its predicted)
    rows.append(res)
names = list(rows[0].keys())
print(f"{len(rows)} cubes; share of kept in the first f of the cut level's nodes")
print("order          " + " ".join(f"f={f:<6g}" for f in FS))
for nm in names:
    print(f"{nm:14s} " + " ".join(f"{x:.3f}   " for x in np.mean([r[nm] for r in rows], axis=0)))
