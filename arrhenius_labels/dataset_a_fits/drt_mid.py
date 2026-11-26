# -*- coding: utf-8 -*-
"""Fit-free check: is there a separate peak near 3-4 ms that appears with age (picture i), or does the fast peak lose its
temperature shift (picture ii)? Fixed-lambda DRT, f <= 2 kHz, SOC step 10, cycles 40/240/480, 15/25/35/45 C, two lambdas.
-> fig8_drt_mid.png + printed peak list. py -3.11"""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load, ok, TS, CYC, TAU, drt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.9, 'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.grid': True, 'grid.alpha': 0.25})
KS = [2, 3, 4, 5]; CYS = [40, 240, 480]; COL = ['#1f5fa8', '#7fa6d6', '#e3a35b', '#c26a12']
def peaks(g):
    i = [j for j in range(1, len(g) - 1) if g[j] >= g[j - 1] and g[j] >= g[j + 1] and g[j] > 0.05 * g.max() and TAU[j] < 1.0]
    return [(TAU[j], g[j]) for j in i]
f, axs = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True)
for lam, ls in ((0.03, '-'), (0.3, ':')):
    for c in (0, 1):
        for ci, n in enumerate(CYS):
            j = CYC.index(n); ax = axs[c, ci]
            for kk, k in enumerate(KS):
                if not ok(c, j, k): continue
                r0, g, rms = drt(*load(c, j, k, 10), lam=lam)
                ax.semilogx(TAU, g, ls, color=COL[kk], lw=1.2 if ls == '-' else 0.9, label=('%d °C' % TS[k]) if lam == 0.03 else None)
                if lam == 0.03: print('cell%d cyc%3d %2d C  peaks (tau ms : height)  %s' % (c + 1, n, TS[k], '  '.join('%.3g:%.2f' % (t * 1e3, h) for t, h in peaks(g))))
            ax.set_title('cell %d, cycle %d' % (c + 1, n), fontsize=9); ax.set_xlim(2e-5, 1.5); ax.axvspan(2e-3, 6e-3, color='#eeeeee', zorder=0)
for ax in axs[1]: ax.set_xlabel('τ (s)')
for ax in axs[:, 0]: ax.set_ylabel('γ(τ) (mΩ)')
axs[0, 0].legend(fontsize=7.5, frameon=False, title='solid λ=0.03, dotted λ=0.3', title_fontsize=7)
axs[0, 0].text(3.5e-3, axs[0, 0].get_ylim()[1] * 0.92, '2–6 ms', ha='center', fontsize=7, color='#666')
f.tight_layout(); f.savefig('fig8_drt_mid.png', dpi=200, bbox_inches='tight', facecolor='white'); print('saved')
