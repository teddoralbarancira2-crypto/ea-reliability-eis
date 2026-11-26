# -*- coding: utf-8 -*-
"""Paper 4 v10: held-out validation of the misfit-ratio rule on dataset A (fig3_rows.json, 44 cases = 2 cells x 3 V x checkpoints).
(1) leave-one-cell-out and leave-one-voltage-out: the threshold t is LEARNED on the training part (largest ratio such that every
training case at or below it has a window span <= tau), then applied to the held-out part. (2) Spearman per cell and per voltage.
(3) clustered statistics: within-cluster permutation test and cluster bootstrap of the pooled Spearman (clusters = cell x voltage).
Writes holdout_v10.json. py -3.11"""
import json, os, sys, itertools, numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, 'fig3_rows.json')))
KEY = sys.argv[1] if len(sys.argv) > 1 else 'ratio_hf'
x = np.array([r[KEY] for r in R]); y = np.array([r['span'] for r in R])
cell = np.array([r['cell'] for r in R]); volt = np.array([r['v'] for r in R]); cyc = np.array([r['cyc'] for r in R])
clus = np.array(['%d_%.2f' % (c, v) for c, v in zip(cell, volt)])
out = {}

def learn(xt, yt, tau):
    o = np.argsort(xt); t = None
    for i in o:
        if yt[i] <= tau: t = xt[i]
        else: break
    return t

def score(xs, ys, t, tau):
    if t is None: return dict(t=None, n=len(xs), n_pass=0)
    p = xs <= t
    return dict(t=round(float(t), 4), n=int(len(xs)), n_pass=int(p.sum()), false_pass=int((p & (ys > tau)).sum()),
                missed_stable=int((~p & (ys <= tau)).sum()), n_stable=int((ys <= tau).sum()),
                max_span_pass=round(float(ys[p].max()), 3) if p.any() else None,
                median_span_fail=round(float(np.median(ys[~p])), 3) if (~p).any() else None)

print('pooled: N %d, spearman %.3f' % (len(x), spearmanr(x, y)[0]))
for tau in (0.08, 0.10, 0.15):
    print('\n=== tau = %.2f eV (a window-stable Ea = spread over the 5 windows <= tau)' % tau)
    t_all = learn(x, y, tau); print('  learned on all 44: t = %.4f' % t_all, score(x, y, t_all, tau))
    for name, groups in (('cell', cell), ('voltage', volt)):
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            t = learn(x[tr], y[tr], tau); s = score(x[te], y[te], t, tau)
            out['%s_out|%s|%.2f' % (name, g, tau)] = s
            print('  leave %s %-5s out: learned t = %s -> held-out %s' % (name, g, s['t'], {k: v for k, v in s.items() if k != 't'}))
    s = score(x, y, 0.015, tau); out['fixed0.015|%.2f' % tau] = s; print('  paper threshold 0.015 on all:', s)
    for c in (1, 2):
        m = cell == c; s = score(x[m], y[m], 0.015, tau); out['fixed0.015|cell%d|%.2f' % (c, tau)] = s; print('    cell %d:' % c, s)
print('\n=== Spearman per group')
for name, groups in (('cell', cell), ('voltage', volt)):
    for g in np.unique(groups):
        m = groups == g; r_, p_ = spearmanr(x[m], y[m]); out['spearman|%s|%s' % (name, g)] = [round(float(r_), 3), float(p_), int(m.sum())]
        print('  %s %-5s N %2d  rho %.3f  p %.1e' % (name, g, m.sum(), r_, p_))
# clustered statistics
rng = np.random.default_rng(0); r0 = spearmanr(x, y)[0]
perm = []
for _ in range(20000):
    yp = y.copy()
    for c in np.unique(clus):
        ix = np.where(clus == c)[0]; yp[ix] = y[rng.permutation(ix)]
    perm.append(spearmanr(x, yp)[0])
p_perm = (np.sum(np.array(perm) >= r0) + 1) / (len(perm) + 1)
U = np.unique(clus); boot = []
for _ in range(20000):
    pick = rng.choice(U, len(U), replace=True); ix = np.concatenate([np.where(clus == c)[0] for c in pick])
    if len(np.unique(x[ix])) > 2: boot.append(spearmanr(x[ix], y[ix])[0])
ci = np.percentile(boot, [2.5, 97.5])
# mean within-cluster spearman
wc = [spearmanr(x[clus == c], y[clus == c])[0] for c in U]
out['clustered'] = dict(spearman=round(float(r0), 3), p_within_cluster_perm=float(p_perm), boot95=[round(float(c), 3) for c in ci],
                        within_cluster_rho={c: round(float(w), 3) for c, w in zip(U, wc)}, n_clusters=int(len(U)))
print('\n=== clustered: pooled rho %.3f | within-cluster permutation p = %.4f | cluster bootstrap 95%% CI [%.2f, %.2f]' % (r0, p_perm, *ci))
print('   within-cluster rho:', {c: round(float(w), 2) for c, w in zip(U, wc)})
json.dump(out, open(os.path.join(H, 'holdout_v10_%s.json' % KEY), 'w'), indent=1); print('saved holdout_v10.json')
