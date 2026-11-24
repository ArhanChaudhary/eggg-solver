#!/usr/bin/env python3
"""Expected improvements (E) of long runs, from their EGGG_DEBUG traces.

    scripts/erate.py [--from S] TRACE...           per position and per time window
    scripts/erate.py [--from S] --pair A.trace B.trace   paired comparison (same positions)

A long run improves only a few times (20 positions x 600 s: ~5 improvements after 10 s), so
comparing final lengths cannot detect effects under ~2x. Instead, every phase-2 solution
(reduction) is a draw: it leads to a shorter solution if its phase 3 (an optimal 3x3 solve,
distributed like the distance of a random 3x3 position) has at most best - 1 - L12 moves. So
a round's chance of an improvement is

    E = sum over its phase-2 levels L12 of  seen(L12) x P(3x3 distance <= best - 1 - L12)

and E summed over a run is the number of improvements the model expects. It resolves
differences of ~10-20% between builds where the lengths need ~2x.

Trace (stderr of `EGGG_DEBUG=1 eggg-solver bench ...` or `solve`): lines "[  t] MSG" with t
the seconds since the position's start; positions follow each other (t starts again near 0).
Used: "P2LEVEL <L12>: ... seen <N> best <B>" (N phase-2 solutions of total length L12 at
best length B; -1 before the first solution, not counted) and "IMPROVE <B> -> <B'> ..."
(observed improvements; the first solution, from -1, is not one).

Calibration: improvements observed after 10 s were 0.5-0.7 of E (x86, 48 threads: 20 x 600 s
runs 6 observed / 12.1 and 6 / 12.6 expected; 20 x 300 s runs 0.63 pooled; the analysis of
2026-11-24 used 0.69). E counts every successful reduction, but successes cluster (the
reductions of one candidate are correlated, and several successes in one round make one
improvement), so read E as a relative measure: compare E between runs, and scale it by
~0.6-0.7 for the number of improvements. Before ~10 s the bests fall every few rounds and E
is far above the improvements seen (each round has many successes).

Pair mode compares, per position and best length, the E each run gathered while BOTH had that
best, over the same wall-clock interval (a run that improves sooner faces a harder target
afterwards; comparing at equal best and equal time removes that). Totals, the ratio B/A, the
positions where B gathered more, and a bootstrap interval over positions.
"""
import random
import re
import sys

# Number of 3x3x3 positions at each distance (half-turn metric, outer face turns: the metric
# of phase 3) from: T. Rokicki, H. Kociemba, M. Davidson, J. Dethridge, "The diameter of the
# Rubik's cube group is twenty", SIAM J. Discrete Math. 27 (2013) and cube20.org (distances
# 0-15 exact, 16-20 approximate; 18 here is the remainder of the total, 2.86e19 for ~2.9e19).
COUNTS = [1, 18, 243, 3240, 43239, 574908, 7618438, 100803036, 1332343288, 17596479795,
          232248063316, 3063288809012, 40374425656248, 531653418284628, 6989320578825358,
          91365146187124313, 1_100_000_000_000_000_000, 12_000_000_000_000_000_000, None,
          1_500_000_000_000_000_000, 490_000_000]
TOTAL = 43_252_003_274_489_856_000
COUNTS[18] = TOTAL - sum(c for c in COUNTS if c is not None)
CUM = []
_s = 0
for _c in COUNTS:
    _s += _c
    CUM.append(_s / TOTAL)


def p_at_most(y):
    """P(optimal 3x3 distance <= y) for a random position (P(<= 17) = 0.305, <= 18 = 0.965)."""
    return 0.0 if y < 0 else CUM[min(y, len(CUM) - 1)]


LINE = re.compile(r'^\[\s*([\d.]+)\] (.*)$')
P2LEVEL = re.compile(r'^P2LEVEL (\d+):.* seen (\d+) best (-?\d+)')
IMPROVE = re.compile(r'^IMPROVE (-?\d+) -> (\d+)')


def positions(path):
    """Per position: (end time, [(t, 'E', best, e)], [(t, old, new)] improvements)."""
    out, cur, last = [], None, -1.0
    for line in open(path, errors='replace'):
        m = LINE.match(line.rstrip('\n'))
        if not m:
            continue
        t, msg = float(m.group(1)), m.group(2)
        if cur is None or t < last - 0.5:
            cur = [0.0, [], []]
            out.append(cur)
        last = t
        cur[0] = max(cur[0], t)
        m = P2LEVEL.match(msg)
        if m:
            l12, seen, best = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if best >= 0 and seen:
                cur[1].append((t, best, seen * p_at_most(best - 1 - l12)))
            continue
        m = IMPROVE.match(msg)
        if m and int(m.group(1)) >= 0:
            cur[2].append((t, int(m.group(1)), int(m.group(2))))
    return [tuple(p) for p in out]


def final_best(pos):
    imps = pos[2]
    return imps[-1][2] if imps else None


def best_intervals(pos, t0, t1):
    """[(best, start, end)] of the position's best length over [t0, t1] (from its E lines and
    improvements: the best before an improvement is its 'old' length)."""
    imps = [(t, old, new) for t, old, new in pos[2]]
    first = next((b for t, b, _ in pos[1]), None)
    if imps:
        first = imps[0][1]
    if first is None:
        return []
    out, cur, start = [], first, 0.0
    for t, old, new in imps:
        out.append((cur, start, t))
        cur, start = new, t
    out.append((cur, start, float('inf')))
    res = []
    for b, s, e in out:
        s, e = max(s, t0), min(e, t1)
        if e > s:
            res.append((b, s, e))
    return res


def e_in(pos, best, s, e):
    return sum(x for t, b, x in pos[1] if b == best and s <= t < e)


def fmt(x):
    return f'{x:.3g}' if x else '0'


def single(paths, t0, windows):
    tot_e = tot_o = 0.0
    for path in paths:
        ps = positions(path)
        print(f'{path}: {len(ps)} positions; E from {t0:g} s')
        print('| pos | final | at (s) | E | observed | E/s, last half | ')
        print('|---:|---:|---:|---:|---:|---:|')
        W = [w for w in windows]
        win_e = [0.0] * (len(W) - 1)
        win_o = [0] * (len(W) - 1)
        fe = fo = 0.0
        for i, p in enumerate(ps):
            end = p[0]
            e = sum(x for t, b, x in p[1] if t >= t0)
            o = sum(1 for t, *_ in p[2] if t >= t0)
            half = sum(x for t, b, x in p[1] if t >= end / 2) / max(end / 2, 1e-9)
            for t, b, x in p[1]:
                for k in range(len(W) - 1):
                    if W[k] <= t < W[k + 1]:
                        win_e[k] += x
            for t, *_ in p[2]:
                for k in range(len(W) - 1):
                    if W[k] <= t < W[k + 1]:
                        win_o[k] += 1
            at = p[2][-1][0] if p[2] else 0.0
            print(f'| {i} | {final_best(p)} | {at:.1f} | {fmt(e)} | {o} | {half:.2e} |')
            fe += e
            fo += o
        print(f'total from {t0:g} s: E {fe:.3f}, observed {fo:.0f}'
              + (f' (observed/E {fo / fe:.2f})' if fe else ''))
        print('| window (s) | ' + ' | '.join(f'{W[k]:g}-{W[k + 1]:g}' for k in range(len(W) - 1)) + ' |')
        print('|---|' + '---:|' * (len(W) - 1))
        print('| E | ' + ' | '.join(f'{x:.3f}' for x in win_e) + ' |')
        print('| observed | ' + ' | '.join(str(x) for x in win_o) + ' |')
        print()
        tot_e += fe
        tot_o += fo
    if len(paths) > 1 and tot_e:
        print(f'all files from {t0:g} s: E {tot_e:.3f}, observed {tot_o:.0f} (observed/E {tot_o / tot_e:.2f})')


def pair(pa, pb, t0, t1, boot):
    A, B = positions(pa), positions(pb)
    assert len(A) == len(B), f'not paired: {len(A)} vs {len(B)} positions'
    rows = []
    for i, (a, b) in enumerate(zip(A, B)):
        end = min(a[0], b[0], t1)
        ia, ib = best_intervals(a, t0, end), best_intervals(b, t0, end)
        ea = eb = ov = 0.0
        for ba, sa, xa in ia:
            for bb, sb, xb in ib:
                if ba != bb:
                    continue
                s, e = max(sa, sb), min(xa, xb)
                if e > s:
                    ov += e - s
                    ea += e_in(a, ba, s, e)
                    eb += e_in(b, bb, s, e)
        oa = sum(1 for t, *_ in a[2] if t0 <= t < end)
        ob = sum(1 for t, *_ in b[2] if t0 <= t < end)
        rows.append((i, final_best(a), final_best(b), ov, ea, eb, oa, ob))
    na, nb = pa.split('/')[-1], pb.split('/')[-1]
    print(f'A = {pa}, B = {pb}: E while both have the same best, from {t0:g} s')
    print(f'| pos | final A | final B | same best (s) | E A | E B | B/A | improvements A | B |')
    print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i, fa, fb, ov, ea, eb, oa, ob in rows:
        r = f'{eb / ea:.2f}' if ea > 0 else '-'
        print(f'| {i} | {fa} | {fb} | {ov:.0f} | {fmt(ea)} | {fmt(eb)} | {r} | {oa} | {ob} |')
    sa = sum(r[4] for r in rows)
    sb = sum(r[5] for r in rows)
    more = sum(1 for r in rows if r[5] > r[4])
    less = sum(1 for r in rows if r[5] < r[4])
    print(f'\nE A {sa:.3f}, E B {sb:.3f}: B/A = {sb / sa:.3f}' if sa else '\nno E in common', end='')
    if sa:
        rnd = random.Random(1)
        rs = []
        for _ in range(boot):
            s = [rows[rnd.randrange(len(rows))] for _ in rows]
            x = sum(r[4] for r in s)
            if x > 0:
                rs.append(sum(r[5] for r in s) / x)
        rs.sort()
        lo, hi = rs[int(0.05 * len(rs))], rs[int(0.95 * len(rs)) - 1]
        print(f' (90% bootstrap over positions {lo:.3f}..{hi:.3f}); B more in {more}, less in {less} of {len(rows)} positions')
    print(f'improvements from {t0:g} s: A {sum(r[6] for r in rows)}, B {sum(r[7] for r in rows)}')


def main():
    args = sys.argv[1:]
    t0, t1, boot = 10.0, float('inf'), 2000
    windows = [0, 10, 30, 60, 120, 300, 600, 1200, 3600, float('inf')]
    if '--from' in args:
        k = args.index('--from')
        t0 = float(args[k + 1])
        del args[k:k + 2]
    if '--to' in args:
        k = args.index('--to')
        t1 = float(args[k + 1])
        del args[k:k + 2]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        return
    if args[0] == '--pair':
        assert len(args) == 3, 'usage: erate.py [--from S] [--to S] --pair A.trace B.trace'
        pair(args[1], args[2], t0, t1, boot)
    else:
        single(args, t0, windows)


if __name__ == '__main__':
    main()
