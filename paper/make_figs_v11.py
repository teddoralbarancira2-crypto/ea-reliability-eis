# -*- coding: utf-8 -*-
"""Figs. 2 to 5 of the paper from the stored analysis results; prints every number shown and checks text overlaps. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import os, sys, json, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib, matplotlib.patches; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), 'figures')
AL = config.AL; AU = os.path.join(AL, 'dataset_a_fits'); ID = os.path.join(AL, 'ident')
ld = lambda p: json.load(open(p)); kB = 8.617333e-5
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.6, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.minor.size': 1.5,
                     'ytick.minor.size': 1.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'legend.fontsize': 6.5, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:bold', 'lines.solid_capstyle': 'round'})
INK, GRY, LGRY = '#1a1a1a', '#8a8a8a', '#c8c8c8'
SLOW, FAST, RED = '#1f4e9a', '#8a8a8a', '#b03a2e'
VC = {4.06: '#0072B2', 3.73: '#009E73', 3.41: '#D55E00'}
def lab(ax, s, x=-0.2): ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
def check(f, n):
    f.canvas.draw(); r = f.canvas.get_renderer(); items = list(f.texts)
    for a in f.axes:
        lo, hi = sorted(a.get_xlim()); lo2, hi2 = sorted(a.get_ylim())
        items += a.texts + [a.xaxis.label, a.yaxis.label, a.title]
        items += [t for l_, t in zip(a.xaxis.get_majorticklocs(), a.xaxis.get_majorticklabels()) if lo <= l_ <= hi]
        items += [t for l_, t in zip(a.yaxis.get_majorticklocs(), a.yaxis.get_majorticklabels()) if lo2 <= l_ <= hi2]
        if a.get_legend(): items.append(a.get_legend())
    bb = [(t.get_text() if hasattr(t, 'get_text') else 'legend', t.get_window_extent(r)) for t in items
          if t.get_visible() and (not hasattr(t, 'get_text') or t.get_text().strip())]
    bad = [(a[0][:25], b[0][:25]) for i, a in enumerate(bb) for b in bb[i + 1:] if a[1].overlaps(b[1])]
    print('CHECK', n, len(bb), 'texts, overlaps:', bad)
def save(f, n): check(f, n); f.savefig(os.path.join(OUT, n + '_v11.png'), dpi=600, bbox_inches='tight', facecolor='white'); plt.close(f); print('saved', n)

# ================= Fig 2: standard criteria accept a window-dependent Ea =================
C = ld(os.path.join(ID, 'curvature.json')); CI = ld(os.path.join(HERE, 'ci_dataset_b.json')); RB = ld(os.path.join(AL, 'robust.json'))
WB = [(0, 20), (5, 25), (10, 30), (15, 35), (20, 40), (25, 45)]; MID = np.array([(a + b) / 2 for a, b in WB])
CASES = [('no1_ch3_50', 'cell 1, 50 % SOC'), ('no1_ch3_70', 'cell 1, 70 % SOC'), ('no2_ch4_50', 'cell 2, 50 % SOC'), ('no2_ch4_70', 'cell 2, 70 % SOC')]
f = plt.figure(figsize=(7.48, 2.45)); gs = f.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.8], wspace=0.42)
a1, a2, a3 = [f.add_subplot(gs[0, i]) for i in range(3)]
d = CI['no1_ch3_50']['slow']; lo, hi = d['Ea'] - 1.96 * d['se'], d['Ea'] + 1.96 * d['se']
a1.axhspan(lo, hi, color='#dce6f2', lw=0, zorder=0); a1.axhline(d['Ea'], color=SLOW, lw=0.6, ls=(0, (3, 2)), zorder=1)
for i, (cs, nm) in enumerate(CASES):
    main = i == 0
    for proc, col in (('slow', SLOW), ('fast', FAST)):
        y = np.array([C['%s_%d-%d' % (cs, t0, t1)][proc] for t0, t1 in WB])
        a1.plot(MID, y, '-', color=col, lw=1.4 if main else 0.6, alpha=1 if main else 0.5, zorder=3 if main else 2,
                marker='o' if main else None, ms=3.2, mfc='white' if proc == 'fast' else col, mec=col, mew=0.8)
        if main: print('Fig2a', proc, dict(zip(MID.tolist(), np.round(y, 3).tolist())))
    print('Fig2a', cs, 'slow CI %.3f-%.3f' % (CI[cs]['slow']['Ea'] - 1.96 * CI[cs]['slow']['se'], CI[cs]['slow']['Ea'] + 1.96 * CI[cs]['slow']['se']))
a1.text(8.6, 0.70, 'slow', color=SLOW, fontsize=6.8, ha='left'); a1.text(8.6, 0.385, 'fast', color=GRY, fontsize=6.8, ha='left')
a1.set_xlim(8, 37); a1.set_xticks(MID); a1.set_xticklabels(['%d–%d' % w for w in WB], rotation=0, fontsize=6.3)
a1.set_xlabel('20 °C temperature window (°C)'); a1.set_ylabel('$E_{a,R}$ (eV)'); a1.set_ylim(0.36, 0.77)
a1.legend(handles=[Line2D([], [], color=INK, lw=1.4, marker='o', ms=3, label='cell 1, 50 % SOC'), Line2D([], [], color=INK, lw=0.6, alpha=0.5, label='other cell or SOC'),
                   matplotlib.patches.Patch(color='#dce6f2', label='95 % CI, full range'), Line2D([], [], color=SLOW, lw=0.6, ls=(0, (3, 2)), label='full-range value')],
          loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=2.0, columnspacing=1.0, fontsize=6.0)
a1.set_clip_on(False)
TB = np.array([-10, 0, 10, 30, 50]) + 273.15
RCT = {1: [203.9, 180.3, 151.2, 41.6, 9.8], 2: [196.6, 178.3, 113.0, 39.8, 9.9], 3: [179.1, 125.4, 63.0, 38.8, 10.2], 4: [190.3, 194.1, 150.9, 42.9, 10.8]}
x = 1 / (kB * TB); CC = {1: '#0072B2', 2: '#D55E00', 3: '#999999', 4: '#009E73'}
def afit(xx, yy):
    p = np.polyfit(xx, yy, 1); r2 = 1 - np.sum((yy - np.polyval(p, xx)) ** 2) / np.sum((yy - yy.mean()) ** 2); return p, r2
res = {}
for c, r in RCT.items():
    y = np.log(r); pf, r2f = afit(x, y); pc, _ = afit(x[:3], y[:3]); pw, r2w = afit(x[2:], y[2:]); res[c] = (pf[0], r2f, pc[0], pw[0], r2w)
    a2.plot(1000 / TB, y, 'o', ms=3.2, color=CC[c], mfc='white' if c == 3 else CC[c], mew=0.8, zorder=3)
    xs = np.linspace(x.min(), x.max(), 50); a2.plot(1000 * kB * xs, np.polyval(pf, xs), '-', color=CC[c], lw=0.7, alpha=0.8)
    print('Fig2b cell %d full Ea %.3f R2 %.3f | cold %.3f | warm %.3f R2 %.3f | diff %.3f' % (c, pf[0], r2f, pc[0], pw[0], r2w, pw[0] - pc[0]))
a2.set_xlabel('1000 / $T$ (K$^{-1}$)'); a2.set_ylabel('ln($R_{ct}$ / mΩ)')
a2.set_xlim(3.05, 3.85); a2.text(0.97, 0.03, 'one fit over −10 to 50 °C:\n$R^2$ = %.2f–%.2f' % (min(v[1] for v in res.values()), max(v[1] for v in res.values())),
                                transform=a2.transAxes, va='bottom', ha='right', fontsize=6.3, color=INK)
a2.text(0.03, 0.97, 'data: Bao and Gong (2023)\ncolors as in c', transform=a2.transAxes, va='top', fontsize=6.0, color=GRY)
for k, c in enumerate(RCT):
    full, r2f, cold, warm, r2w = res[c]; yy = 4 - k
    a3.plot([cold, warm], [yy, yy], '-', color=LGRY, lw=2.2, zorder=1)
    a3.plot(cold, yy, 'o', ms=4, mfc='white', mec=CC[c], mew=1.0, zorder=3); a3.plot(warm, yy, 'o', ms=4, color=CC[c], zorder=3)
    a3.plot(full, yy, '|', ms=7, mew=1.2, color=INK, zorder=4); a3.text(0.6, yy, '$R^2$ %.2f' % r2f, fontsize=6.0, va='center', ha='left', color=GRY)
a3.set_yticks([4, 3, 2, 1]); a3.set_yticklabels(['cell %d' % c for c in RCT]); a3.set_ylim(0.4, 4.9); a3.set_xlim(0, 0.6); a3.set_xticks([0, 0.2, 0.4, 0.6])
a3.set_xlabel('$E_{a,R}$ (eV)'); a3.spines['left'].set_visible(False); a3.tick_params(axis='y', length=0)
a3.legend(handles=[Line2D([], [], marker='o', lw=0, mfc='white', mec=INK, ms=4, label='−10 to 10 °C'), Line2D([], [], marker='o', lw=0, color=INK, ms=4, label='10 to 50 °C'),
                   Line2D([], [], marker='|', lw=0, color=INK, ms=7, mew=1.2, label='full range')], loc='lower center', bbox_to_anchor=(0.62, 1.0), ncol=3,
          handletextpad=0.2, columnspacing=0.7, fontsize=6.0)
lab(a1, 'a', -0.17); lab(a2, 'b', -0.25); lab(a3, 'c', -0.3); save(f, 'fig2')

# ================= Fig 3: synthetic tests =================
S = ld(os.path.join(ID, 'synth_ident.json')); TP = ld(os.path.join(ID, 'synth_transplant.json'))
med = lambda d, k: float(np.median([v[0] for v in d[k]]))
def series(src, r=None, m=None, **flt):
    pts = [d for d in src if all(d[k] == v for k, v in flt.items()) and (r is None or d['r'] == r) and (m is None or d['m'] == m)]
    key = 'm' if r is not None else 'r'; pts.sort(key=lambda d: d[key])
    return np.array([d[key] for d in pts]), np.array([abs(med(d, 'drop0') - med(d, 'full')) for d in pts])
SER = [(S, dict(mode='W', sigma=0.06), 'noise only', '#9a9a9a', 'o', '-'), (S, dict(mode='M2', sigma=0.06), 'smooth misspecification', '#C99A1E', 's', '-'),
       (TP, dict(where='cold'), 'real residual at 0 and 5 °C', '#009E73', '^', (0, (3, 1.5))), (TP, dict(where='all'), 'real residual at all temperatures', RED, 'D', '-')]
f = plt.figure(figsize=(7.48, 2.45)); gs3 = f.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.12)
a1 = f.add_subplot(gs3[0, 0]); a2 = f.add_subplot(gs3[0, 1], sharey=a1); a3 = f.add_subplot(gs3[0, 2]); plt.setp(a2.get_yticklabels(), visible=False)
gs3.update(wspace=0.12); a3.set_position([a3.get_position().x0 + 0.075, a3.get_position().y0, a3.get_position().width - 0.075, a3.get_position().height])
jk = [max(RB[c]['jk_EaR_s']) - RB[c]['full']['EaR_s'] for c, _ in CASES]; print('Fig3 measured drop-0 swing', np.round(jk, 3))
for ax, fixed in ((a1, dict(r=7)), (a2, dict(m=0.3))):
    ax.axhspan(min(jk), max(jk), color='#f4e4e1', lw=0, zorder=0)
    for src, flt, nm, col, mk, ls in SER:
        xx, yy = series(src, **fixed, **flt); ax.plot(xx, yy, ls=ls, marker=mk, ms=3.4, color=col, lw=1.1, mec=col, mfc=col if mk != 's' else 'white', label=nm)
        pts = sorted([d for d in src if all(d[k] == v for k, v in flt.items()) and all(d[k] == v for k, v in fixed.items())], key=lambda d: d['m' if 'r' in fixed else 'r'])
        sw = [np.abs(np.array([v[0] for v in d['drop0']]) - np.array([v[0] for v in d['full']])) for d in pts]
        ax.vlines(xx, [w.min() for w in sw], [w.max() for w in sw], color=col, lw=0.7, alpha=0.6, zorder=1)
        print('Fig3 spread', fixed, nm, [(round(float(w.min()), 3), round(float(w.max()), 3), len(w)) for w in sw])
        print('Fig3', fixed, nm, list(zip(xx, np.round(yy, 3))))
    ax.set_xscale('log'); ax.set_ylim(-0.008, 0.27); ax.minorticks_off()
a1.set_ylabel('shift of slow $E_{a,R}$ without 0 °C (eV)')
a1.set_xlabel('size ratio $R_{slow}$ / $R_{fast}$\n(time-constant ratio 7)'); a1.set_xticks([0.3, 1, 3, 10]); a1.set_xticklabels(['0.3', '1', '3', '10'])
a2.set_xlabel('time-constant ratio $τ_{slow}$ / $τ_{fast}$\n(size ratio 0.3)'); a2.set_xticks([3, 10, 30, 100]); a2.set_xticklabels(['3', '10', '30', '100'])
a2.text(100, max(jk) - 0.006, 'measured on dataset B', fontsize=6.3, color=RED, ha='right', va='top')
a1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4, handlelength=2.2, columnspacing=1.0, fontsize=6.0)
RMB = ld(os.path.join(HERE, 'analysis_v13.json'))['resid_B']['no1_ch3_50']; fr = np.array(RMB['f'])
import matplotlib.cm as cm
for i, T in enumerate(RMB['T']):
    col = cm.coolwarm((T - 0) / 45); a3.semilogx(fr, RMB['re'][i], color=col, lw=0.9 if T in (0, 5) else 0.6, alpha=1 if T in (0, 5) else 0.75)
a3.axhline(0, color=GRY, lw=0.5); a3.set_xlabel('frequency (Hz)'); a3.set_ylabel('residual, real part (mΩ)', labelpad=1)
a3.text(0.97, 0.95, '0 °C', transform=a3.transAxes, color=cm.coolwarm(0.0), fontsize=6.3, ha='right', va='top')
a3.text(0.97, 0.86, '45 °C', transform=a3.transAxes, color=cm.coolwarm(1.0), fontsize=6.3, ha='right', va='top')
print('Fig3c residual rms per T', dict(zip(RMB['T'], np.round(RMB['rms_T'], 3))))
lab(a1, 'a', -0.17); lab(a2, 'b', -0.05); lab(a3, 'c', -0.3)
for a_ in (a1, a2, a3): a_.texts[-1].set_y(1.13)
save(f, 'fig3')

# ================= Fig 4: misfit-to-arc ratio vs window span =================
rows = ld(os.path.join(HERE, 'fig3_rows.json'))
f, ax = plt.subplots(figsize=(3.54, 2.9))
MK = {3.41: 'v', 3.73: 'o', 4.06: '^'}; cmap = plt.get_cmap('viridis_r'); norm = plt.Normalize(40, 520)
ax.axvspan(0.004, 0.015, color='#e8f1e8', lw=0, zorder=0)
for r in rows:
    ax.scatter(r['ratio_hf'], r['span'], marker=MK[r['v']], s=18, c=[cmap(norm(r['cyc']))], edgecolors=INK, linewidths=0.3, zorder=3)
wb = []
for cs, _ in CASES:
    sp = [C['%s_%d-%d' % (cs, *w)]['slow'] for w in WB] + [CI[cs]['slow']['Ea']]; wb.append((CI[cs]['rms'] / CI[cs]['slow']['R25'], max(sp) - min(sp)))
wb = np.array(wb); print('Fig4 dataset B ratio', np.round(wb[:, 0], 4), 'span', np.round(wb[:, 1], 3))
ax.scatter(wb[:, 0], wb[:, 1], marker='*', s=48, c=RED, edgecolors=INK, linewidths=0.3, zorder=4)
ax.annotate('dataset B', xy=(wb[:, 0].max(), wb[:, 1].max()), xytext=(0.10, 0.62), fontsize=6.5, color=RED, ha='center',
            arrowprops=dict(arrowstyle='-', color=RED, lw=0.5))
ax.axvline(0.015, color=INK, lw=0.7, ls=(0, (4, 2))); ax.axhline(0.072, color=GRY, lw=0.5, ls=':')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.004, 0.14); ax.set_ylim(0.008, 2.0)
ax.set_xlabel('misfit-to-arc ratio $ρ$ = $r_{rms}$ / $R_{slow,25}$'); ax.set_ylabel('window span of slow $E_{a,R}$ (eV)')
ax.text(0.0145, 1.75, 'pass: 26 of 26\nspans ≤ 0.072 eV', ha='right', va='top', fontsize=6.3, color='#2e6b2e')
ax.text(0.0157, 1.75, 'Spearman 0.77 (44 cases)', ha='left', va='top', fontsize=6.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); cb = f.colorbar(sm, ax=ax, pad=0.02, fraction=0.045, ticks=[40, 120, 200, 280, 360, 440, 480])
cb.set_label('cycle (dataset A)', fontsize=6.8); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.5)
ax.legend(handles=[Line2D([], [], marker=MK[v], color='w', mec=INK, mfc='#cccccc', ms=4, mew=0.5, label='%.2f V' % v) for v in (4.06, 3.73, 3.41)],
          loc='lower right', handletextpad=0.2)
save(f, 'fig4')

# ================= Fig 5: what survives on the aging cells =================
sys.path.insert(0, AU); os.chdir(AU)
from common import load, ok, TS, CYC, TAU, drt
Tk = np.array([0, 5, 15, 25, 35, 45, 55]) + 273.15; xT = 1 / (kB * Tk); X = xT - 1 / (kB * 298.15)
f = plt.figure(figsize=(7.48, 2.45)); gs = f.add_gridspec(1, 3, wspace=0.58); a1, a2, a3 = [f.add_subplot(gs[0, i]) for i in range(3)]
SI = {4.06: 3, 3.73: 10, 3.41: 17}; YL = (0.25, 0.85)
a2.axvspan(100, 220, color='#f2f2f2', lw=0, zorder=0)
for v in (4.06, 3.73, 3.41):
    H = ld(os.path.join(AU, 'hf_B_s%d.json' % SI[v])); W = ld(os.path.join(AU, 'window_scan_s%d.json' % SI[v]))
    rr = {(r['cell'], r['cyc']): r for r in rows if r['v'] == v}
    for c, mk in ((1, 'o'), (2, 's')):
        cy, fb, sl, lo_, hi_, ps, fbl, fbh = [], [], [], [], [], [], [], []
        for k, h in H.items():
            if not k.startswith('cell%d_' % c): continue
            n = h['cyc']; q = h['q']; y = sum(np.exp(q[5 * j] + q[5 * j + 1] * X) for j in (0, 1)); cy.append(n); fb.append(np.polyfit(xT, np.log(y), 1)[0])
            wr = [np.polyfit(xT[(Tk - 273.15 >= a_) & (Tk - 273.15 <= b_)], np.log(y[(Tk - 273.15 >= a_) & (Tk - 273.15 <= b_)]), 1)[0] for a_, b_ in ((0, 25), (5, 35), (15, 45), (25, 55))]
            fbl.append(min(wr + [fb[-1]])); fbh.append(max(wr + [fb[-1]]))
            e = [W[kk]['EaR'] for kk in W if kk.startswith('%d_%d_' % (c, n))]
            sl.append(q[11]); lo_.append(min(e) if e else np.nan); hi_.append(max(e) if e else np.nan); ps.append(rr[(c, n)]['ratio_hf'] <= 0.015 if (c, n) in rr else False)
        o = np.argsort(cy); cy, fb, sl, lo_, hi_, ps, fbl, fbh = [np.array(a)[o] for a in (cy, fb, sl, lo_, hi_, ps, fbl, fbh)]
        off = {4.06: -9, 3.73: 0, 3.41: 9}[v] + (c - 1.5) * 4
        a2.plot(cy, fb, '-', marker=mk, ms=2.8, color=VC[v], lw=0.8, mfc=VC[v] if c == 1 else 'white', mew=0.7)
        a2.vlines(cy + (c - 1.5) * 5, fbl, fbh, color=VC[v], lw=0.6, alpha=0.6, zorder=1)
        print('Fig5b %.2f V cell %d window range' % (v, c), {int(a): (round(float(l), 3), round(float(h), 3)) for a, l, h in zip(cy, fbl, fbh)})
        m = cy >= 120
        for xx, yy, l, hgh, p in zip(cy[m], sl[m], lo_[m], hi_[m], ps[m]):
            l2, h2 = max(l, YL[0]), min(hgh, YL[1])
            a1.plot([xx + off] * 2, [l2, h2], '-', color=VC[v], lw=0.6, alpha=0.7, zorder=2)
            if hgh > YL[1]: a1.plot(xx + off, YL[1] - 0.004, marker=6, ms=2.6, color=VC[v], clip_on=False, zorder=2)
            if l < YL[0]: a1.plot(xx + off, YL[0] + 0.004, marker=7, ms=2.6, color=VC[v], clip_on=False, zorder=2)
            a1.plot(xx + off, yy, mk, ms=3.0, color=VC[v], mfc=VC[v] if p else 'white', mew=0.7, zorder=3)
        print('Fig5 %.2f V cell %d' % (v, c), 'fast-band Ea', dict(zip(cy.tolist(), np.round(fb, 3).tolist())), '| slow (pass)', {int(a): round(b, 3) for a, b, p in zip(cy, sl, ps) if p})
a1.set_xlim(95, 505); a2.set_xlim(20, 505)
for a in (a1, a2): a.set_xticks([100, 200, 300, 400, 500]); a.set_xlabel('cycle')
a1.set_ylim(*YL); a1.set_ylabel('slow-process $E_{a,R}$ (eV)')
for v, yy in zip((4.06, 3.73, 3.41), (0.41, 0.535, 0.60)): a1.text(512, yy, '%.2f V' % v, color=VC[v], fontsize=6.3, ha='left', va='center', clip_on=False)
a1.legend(handles=[Line2D([], [], marker='o', lw=0, color=INK, ms=3, label='cell 1'), Line2D([], [], marker='s', lw=0, color=INK, ms=3, label='cell 2'),
                   Line2D([], [], marker='o', lw=0, mfc='white', mec=INK, ms=3, label='ρ > 0.015 (fails)')], loc='lower left', ncol=3, handletextpad=0.1, columnspacing=0.6,
          fontsize=6.0, bbox_to_anchor=(0.0, 1.0))
a2.set_ylabel('fast-band effective $E_a$ (eV)'); a2.set_ylim(0.12, 0.62)
a2.text(160, 0.155, 'slow label\nfails', fontsize=6.0, color=GRY, ha='center', va='bottom')
a2.legend(handles=[Line2D([], [], marker='o', color=INK, lw=0.8, ms=2.8, label='cell 1'), Line2D([], [], marker='s', color=INK, lw=0.8, ms=2.8, mfc='white', label='cell 2')],
          loc='upper right')
for (w0, w1) in ((0.1, 1.2), (1.5, 6)): a3.axvspan(w0, w1, color='#efefef', zorder=0, lw=0)
for n, ls in ((40, '-'), (480, (0, (3, 1.6)))):
    j = CYC.index(n)
    for T, col in ((15, SLOW), (55, RED)):
        k = list(TS).index(T); r0, g, rms = drt(*load(0, j, k, 10), lam=0.03); a3.semilogx(TAU * 1e3, g, color=col, lw=1.0, ls=ls)
        bands = [(np.sum(g[(TAU >= a) & (TAU <= b)]) * np.log(TAU[1] / TAU[0])) for a, b in ((1e-4, 1.2e-3), (1.5e-3, 6e-3))]
        print('Fig5c cell1 cyc %d %d C fast-band area %.2f mid-band %.2f mOhm' % (n, T, *bands))
a3.text(0.35, 4.46, 'fast', fontsize=6.0, color=GRY, ha='center', va='bottom', clip_on=False); a3.text(3.0, 4.46, 'mid', fontsize=6.0, color=GRY, ha='center', va='bottom', clip_on=False)
a3.set_xlim(0.03, 30); a3.set_ylim(-0.08, 4.4); a3.set_xlabel('τ (ms)'); a3.set_ylabel('γ(τ) (mΩ)')
a3.text(0.045, 1.9, '15 °C', color=SLOW, fontsize=6.5); a3.text(0.045, 0.45, '55 °C', color=RED, fontsize=6.5)
a3.text(29, 3.2, 'fast-window area (mΩ)\ncycle 40 → 480\n15 °C: 4.06 → 3.95\n55 °C: 0.50 → 0.86', fontsize=5.8, color=INK, ha='right', va='top')
a3.legend(handles=[Line2D([], [], color=INK, lw=1.0, label='cycle 40'), Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.6)), label='cycle 480')], loc='upper left',
          handlelength=2.0, bbox_to_anchor=(-0.02, 0.92), labelspacing=0.3)
for ax, s in zip((a1, a2, a3), 'abc'): lab(ax, s, -0.27)
save(f, 'fig5')
