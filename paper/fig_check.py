# -*- coding: utf-8 -*-
"""Paper 4: recompute the numbers behind Figs 1-4 from the stored analysis results (no refits except the dataset B Jacobian SE).
py -3.11 fig_check.py"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import json, os, sys, numpy as np
from scipy.stats import spearmanr
AL = config.AL
AU = os.path.join(AL, 'dataset_a_fits'); ID = os.path.join(AL, 'ident')
ld = lambda p: json.load(open(p))
# ---- pooled rule (Fig 3)
WIN = ['0-25', '5-35', '15-45', '25-55', '0-55']
rows = []
for s, v in ((3, 4.06), (10, 3.73), (17, 3.41)):
    W = ld(os.path.join(AU, 'window_scan_s%d.json' % s)); H = ld(os.path.join(AU, 'hf_B_s%d.json' % s))
    keys = sorted(set(k.rsplit('_', 1)[0] for k in W))
    for k in keys:
        e = [W['%s_%s' % (k, w)]['EaR'] for w in WIN if '%s_%s' % (k, w) in W]
        full = W[k + '_0-55']; c, n = k.split('_')
        hb = H.get('cell%s_%s' % (c, n))
        rows.append(dict(v=v, cell=int(c), cyc=int(n), span=max(e) - min(e), nwin=len(e), ratio_ws=full['rms'] / full['R25'],
                         ratio_hf=(hb['rms'] / np.exp(hb['q'][10])) if hb else np.nan, R25=full['R25'], Ea=full['EaR']))
print('N cases', len(rows), 'nwin', set(r['nwin'] for r in rows))
for key in ('ratio_ws', 'ratio_hf'):
    x = np.array([r[key] for r in rows]); y = np.array([r['span'] for r in rows]); m = np.isfinite(x)
    rho, p = spearmanr(x[m], y[m]); lo = m & (x <= 0.015)
    print(key, 'N', m.sum(), 'spearman %.3f p %.1e' % (rho, p), '<=0.015: %d, max span %.3f' % (lo.sum(), y[lo].max()), '>0.015 median %.3f' % np.median(y[m & (x > 0.015)]))
json.dump(rows, open(os.path.join(os.path.dirname(__file__), 'fig3_rows.json'), 'w'), indent=1)
# ---- dataset B windows (Fig 1a)
C = ld(os.path.join(ID, 'curvature.json')); R = ld(os.path.join(AL, 'robust.json'))
for case in ('no1_ch3_50', 'no1_ch3_70', 'no2_ch4_50', 'no2_ch4_70'):
    sw = [C['%s_%s' % (case, w)] for w in ('0-20', '5-25', '10-30', '15-35', '20-40', '25-45')]
    full = C[case + '_0-45']
    print(case, 'slow win %.3f-%.3f' % (min(d['slow'] for d in sw), max(d['slow'] for d in sw)), 'fast win %.3f-%.3f' % (min(d['fast'] for d in sw), max(d['fast'] for d in sw)),
          'jk slow %.3f-%.3f fast %.3f-%.3f' % (min(R[case]['jk_EaR_s']), max(R[case]['jk_EaR_s']), min(R[case]['jk_EaR_f']), max(R[case]['jk_EaR_f'])),
          'ratio %.4f (rms %.4f / slowR %.3f)' % (full['rms'] / full['slowR'], full['rms'], full['slowR']))
# ---- synthetic (Fig 2)
S = ld(os.path.join(ID, 'synth_ident.json')); T = ld(os.path.join(ID, 'synth_transplant.json'))
sw = lambda d, a: abs(np.median([x[0] for x in d[a]]) - np.median([x[0] for x in d['full']]))
for d in S:
    if d['r'] == 7: print('synth', d['mode'], d['sigma'], 'm', d['m'], 'full %.3f drop0 swing %.3f' % (np.median([x[0] for x in d['full']]), sw(d, 'drop0')))
for d in T:
    print('transplant', d['where'], 'r', d['r'], 'm', d['m'], 'full %.3f drop0 %.3f drop5 %.3f' % tuple(np.median([x[0] for x in d[a]]) for a in ('full', 'drop0', 'drop5')))
