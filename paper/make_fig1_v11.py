# -*- coding: utf-8 -*-
"""Fig. 1 of the paper: test sequence, joint model and reliability test, drawn from the dataset A cube and the joint fit of cell 1 at 3.73 V. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.path.dirname(HERE), 'figures')
CUBE = config.CUBE
HF = os.path.join(config.AL, 'dataset_a_fits', 'hf_B_s10.json')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out',
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.labelsize': 6.5,
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'})
kB = 8.617333e-5; NL = chr(10)
HEAD, INK, GRY, BLUE, ORG, RED, DRED = '#1f6fc5', '#222222', '#8c8c8c', '#1f5fa8', '#b5651d', '#b03a2e', '#a1251b'
z = np.load(CUBE); Z, V, F, TS = z['Z'], z['V'], z['f'], np.array(z['T']); CYC = [int(c) for c in z['cyc']]
q = json.load(open(HF))['cell1_480']['q']            # per process [ln R25, EaR, ln tau25, Eatau, alpha]: fast, mid, slow

W, H = 7.2, 4.7
fig = plt.figure(figsize=(W, H))
def rbox(x, y, w, h, fc='white', ec=INK, lw=0.8, ls='-', r=0.008, z=1):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=%g' % r, transform=fig.transFigure,
                                      fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
def t(x, y, s, **k): fig.text(x, y, s, **k)
def down(x, y0, y1, label=None):
    fig.patches.append(FancyArrowPatch((x, y0), (x, y1), transform=fig.transFigure, arrowstyle='simple,head_width=5,head_length=4,tail_width=1.6',
                                       color=INK, lw=0, zorder=3))
    if label: t(x + 0.012, (y0 + y1) / 2, label, fontsize=6.6, va='center')
def chevron(x, y):
    w, h = 0.022, 0.06
    fig.patches.append(Polygon([(x, y + h / 2), (x + w * 0.55, y + h / 2), (x + w, y), (x + w * 0.55, y - h / 2), (x, y - h / 2), (x + w * 0.45, y)],
                               closed=True, transform=fig.transFigure, fc='#a9c8ec', ec='#5b8fc9', lw=0.6, zorder=3))
def thumb(rect):
    ax = fig.add_axes(rect); ax.set_zorder(5)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    return ax

COLS = [(0.008, 0.318, 'Experiment'), (0.343, 0.317, 'Model'), (0.677, 0.315, 'Reliability test')]
TOP, BOT, SUMH = 0.905, 0.012, 0.085
for x, w, name in COLS:
    rbox(x, BOT, w, TOP - BOT, fc='white', ec=INK, lw=0.9, ls=(0, (4, 3)), r=0.006, z=0)
    rbox(x + 0.004, TOP - 0.004, w - 0.008, 0.075, fc=HEAD, ec=HEAD, r=0.012, z=2)
    t(x + w / 2, TOP + 0.034, name, ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    fig.lines.append(plt.Line2D([x + 0.006, x + w - 0.006], [BOT + SUMH + 0.008] * 2, transform=fig.transFigure, color=INK, lw=0.7, ls=(0, (4, 3))))
chevron(0.3285, 0.50); chevron(0.6625, 0.50)
SUMY = BOT + SUMH / 2 + 0.004
t(0.167, SUMY, '2 cells × 11 checkpoints' + NL + '× 7 temperatures × 3 voltages', ha='center', va='center', fontsize=7.2, color=DRED, fontweight='bold')
t(0.5015, SUMY, 'One fit over all temperatures gives' + NL + 'each process its activation energies', ha='center', va='center', fontsize=7.2, color=DRED, fontweight='bold')
t(0.8345, SUMY, 'R² and the confidence interval pass labels;' + NL + 'ρ screens their window span', ha='center', va='center', fontsize=7.2, color=DRED, fontweight='bold')

# ============ column 1: Experiment ============
x0, cw = 0.022, 0.290
rbox(x0, 0.715, cw, 0.17)
t(x0 + cw / 2, 0.868, 'Cycle aging, 0.5C, 2.5 to 4.2 V', ha='center', va='center', fontsize=7.6)
ax = thumb([x0 + 0.03, 0.778, cw - 0.06, 0.052])
cy = np.array(CYC); k = cy != 400
ax.plot(cy[k], np.zeros(k.sum()), 'o', ms=3.3, color=BLUE); ax.plot([400], [0], 'x', ms=3.6, color=RED, mew=1.0)
ax.set_xlim(0, 500); ax.set_ylim(-1, 1); ax.set_yticks([]); ax.set_xticks([40, 240, 480]); ax.set_xticklabels(['40', '240', '480']); ax.set_xlabel('cycle', labelpad=1)
for s in ('left', 'right', 'top'): ax.spines[s].set_visible(False)
down(x0 + cw / 2, 0.712, 0.672, 'every 40 cycles')
rbox(x0, 0.565, cw, 0.105)
t(x0 + cw / 2, 0.648, 'Reference test at seven temperatures', ha='center', va='center', fontsize=7.6)
cm = plt.get_cmap('coolwarm')
for i, T in enumerate(TS):
    cx = x0 + 0.010 + i * 0.0375
    rbox(cx, 0.578, 0.032, 0.040, fc=cm(i / 6), ec='none', r=0.006, z=2)
    t(cx + 0.016, 0.598, '%d' % T, ha='center', va='center', fontsize=6.8, color='white' if i in (0, 6) else INK, fontweight='bold', zorder=4)
t(x0 + cw - 0.003, 0.598, '°C', ha='right', va='center', fontsize=6.2)
down(x0 + cw / 2, 0.562, 0.522, 'at each temperature')
rbox(x0, 0.115, cw, 0.405)
t(x0 + cw / 2, 0.503, 'Impedance test', ha='center', va='center', fontsize=7.6)
ax = thumb([x0 + 0.045, 0.335, cw - 0.065, 0.120])
soc = 100 - 5 * np.arange(21); v = V[0, 0, 3]
ax.step(soc, v, where='post', color=INK, lw=0.7); ax.plot(soc, v, 'o', ms=1.5, color=GRY)
for s_, c in [(3, '#0072B2'), (10, '#009E73'), (17, '#D55E00')]:
    ax.plot(soc[s_], v[s_], 'o', ms=3.6, color=c, zorder=4)
    ax.annotate('%.2f V' % v[s_], (soc[s_], v[s_]), xytext={3: (6, -13), 10: (6, 3), 17: (-4, -9)}[s_], textcoords='offset points', ha='right' if s_ == 17 else 'left', fontsize=5.8, color=c)
ax.set_xlim(102, -2); ax.set_xticks([100, 0]); ax.set_xticklabels(['full', 'empty']); ax.set_ylabel('rest V (V)', labelpad=1)
ax.set_title('21 steps of 5 %, 1 h rest, then EIS', fontsize=6.5, pad=2)
ax = thumb([x0 + 0.045, 0.172, cw - 0.065, 0.095])
j = CYC.index(480)
for i, T in enumerate(TS):
    zz = Z[0, j, i, 10] * 1e3; ax.plot(zz.real, -zz.imag, lw=0.8, color=cm(i / 6))
ax.text(118, 14, '0 °C', fontsize=5.8, color=cm(0.0), ha='right', va='bottom'); ax.text(118, 3, '55 °C', fontsize=5.8, color=cm(1.0), ha='right', va='bottom')
ax.set_xlim(22, 120); ax.set_ylim(-3, 42); ax.set_xlabel("Z' (mΩ)", labelpad=2); ax.set_ylabel("−Z'' (mΩ)", labelpad=1)
ax.set_title('10 kHz to 0.1 Hz (cycle 480, 3.73 V)', fontsize=6.5, pad=2)

# ============ column 2: Model ============
x0, cw = 0.357, 0.289
rbox(x0, 0.635, cw, 0.25)
t(x0 + cw / 2, 0.868, 'Equivalent circuit', ha='center', va='center', fontsize=7.6)
ax = fig.add_axes([x0 + 0.008, 0.675, cw - 0.016, 0.17]); ax.set_zorder(5); ax.set_xlim(0, 10); ax.set_ylim(-1.6, 1.6); ax.axis('off')
def res(x, y, w=0.7, h=0.32, c=INK): ax.add_patch(plt.Rectangle((x, y - h / 2), w, h, fc='white', ec=c, lw=0.9, zorder=3))
def cpe(x, y, c=INK):
    for dx in (-0.1, 0.12): ax.plot([x + dx - 0.1, x + dx + 0.08, x + dx - 0.1], [y + 0.3, y, y - 0.3], color=c, lw=1.0, zorder=5)
ax.plot([0.1, 9.9], [0, 0], color=INK, lw=0.9, zorder=1)
for xi in np.arange(0.35, 1.25, 0.22): ax.add_patch(matplotlib.patches.Arc((xi, 0), 0.22, 0.45, theta1=0, theta2=180, lw=0.9, color=INK, zorder=3))
ax.add_patch(plt.Rectangle((0.3, -0.05), 0.95, 0.12, fc='white', ec='none', zorder=2))
for xi in np.arange(0.35, 1.25, 0.22): ax.add_patch(matplotlib.patches.Arc((xi, 0.0), 0.22, 0.45, theta1=0, theta2=180, lw=0.9, color=INK, zorder=4))
ax.text(0.78, -0.55, 'L', ha='center', fontsize=6.6); res(1.6, 0, c=INK); ax.text(1.95, -0.55, '$R_0$', ha='center', fontsize=6.6)
for i, (xc, nm, c) in enumerate([(3.4, 'fast', GRY), (5.5, 'mid', GRY), (7.6, 'slow', BLUE)]):
    ax.add_patch(plt.Rectangle((xc - 0.25, -0.75), 1.2, 1.5, fc='white', ec='white', zorder=2))
    ax.plot([xc - 0.25, xc - 0.25], [-0.5, 0.5], color=c, lw=0.9, zorder=3); ax.plot([xc + 0.95, xc + 0.95], [-0.5, 0.5], color=c, lw=0.9, zorder=3)
    ax.plot([xc - 0.25, xc + 0.95], [0.5, 0.5], color=c, lw=0.9, zorder=3); ax.plot([xc - 0.25, xc + 0.95], [-0.5, -0.5], color=c, lw=0.9, zorder=3)
    res(xc + 0.0, 0.5, c=c); ax.add_patch(plt.Rectangle((xc - 0.24, -0.6), 1.18, 0.2, fc='white', ec='none', zorder=3)); cpe(xc + 0.28, -0.5, c=c)
    ax.text(xc + 0.35, 0.92, '$R_%d$' % (i + 1), ha='center', fontsize=6.6, color=c); ax.text(xc + 0.35, -1.2, nm, ha='center', fontsize=6.4, color=c)
ax.add_patch(plt.Rectangle((9.05, -0.3), 0.5, 0.6, fc='white', ec='none', zorder=2)); cpe(9.25, 0); ax.text(9.3, -0.55, 'tail', ha='center', fontsize=6.4)
t(x0 + cw / 2, 0.652, 'three ZARC processes, constant-phase tail', ha='center', va='center', fontsize=6.4, color='#444')
t(x0 + cw / 2, 0.607, '+', ha='center', va='center', fontsize=15, color=HEAD, fontweight='bold')
rbox(x0, 0.275, cw, 0.305)
t(x0 + cw / 2, 0.566, 'Arrhenius coupling', ha='center', va='center', fontsize=7.6)
t(x0 + cw / 2, 0.538, '$R_k(T) = R_{k,25}\\,\\exp(E_{a,R,k}X)$', ha='center', va='center', fontsize=6.6)
t(x0 + cw / 2, 0.503, '$τ_k(T) = τ_{k,25}\\,\\exp(E_{a,τ,k}X)$', ha='center', va='center', fontsize=6.6)
ax = thumb([x0 + 0.06, 0.335, cw - 0.085, 0.118])
Tg = np.linspace(0, 55, 60) + 273.15; X = 1 / (kB * Tg) - 1 / (kB * 298.15); Xm = 1 / (kB * (TS + 273.15)) - 1 / (kB * 298.15)
Rf = lambda j, x: np.exp(q[5 * j] + q[5 * j + 1] * x)
for lbl, fn, c, xy in [('slow', lambda x: Rf(2, x), BLUE, (0.06, 0.84)), ('fast band', lambda x: Rf(0, x) + Rf(1, x), GRY, (0.40, 0.07))]:
    ax.semilogy(1000 / Tg, fn(X), color=c, lw=1.1); e = np.polyfit(Xm, np.log(fn(Xm)), 1)[0]
    ax.text(*xy, '%s, %.2f eV' % (lbl, e), transform=ax.transAxes, fontsize=6.0, color=c); print(lbl, round(e, 3))
ax.set_xlabel('1000 / T (K$^{-1}$)', labelpad=1); ax.set_ylabel('R (mΩ)', labelpad=1); ax.set_xlim(2.98, 3.68)
top = ax.secondary_xaxis('top', functions=(lambda u: 1000 / np.maximum(u, 1e-9) - 273.15, lambda c: 1000 / (c + 273.15)))
top.set_xticks([0, 25, 55]); top.set_xticklabels(['0 °C', '25 °C', '55 °C']); top.tick_params(labelsize=5.6, pad=1, length=2)
down(x0 + cw / 2, 0.272, 0.232, 'one joint fit')
rbox(x0, 0.118, cw, 0.112, fc='#f3f6fb', ec=HEAD)
t(x0 + cw / 2, 0.212, 'applied to', ha='center', va='center', fontsize=6.6, color='#444')
for i, (nm, sub) in enumerate([('Dataset A', 'aging cells'), ('Dataset B', 'fresh cells'), ('Synthetic', 'spectra')]):
    bx = x0 + 0.010 + i * 0.0935
    rbox(bx, 0.128, 0.085, 0.066, fc='white', ec=HEAD, lw=0.7, r=0.006, z=2)
    t(bx + 0.0425, 0.171, nm, ha='center', va='center', fontsize=6.7, fontweight='bold', zorder=4)
    t(bx + 0.0425, 0.145, sub, ha='center', va='center', fontsize=6.2, zorder=4)

# ============ column 3: Reliability test ============
x0, cw = 0.691, 0.287
t(x0 + cw / 2, 0.868, 'Is the activation energy a property' + NL + 'of the cell or of the fit?', ha='center', va='center', fontsize=7.3, style='italic')
DG = [('Jacobian 95 % confidence interval', 'standard criterion: measures scatter', '#f4f4f4', GRY),
      ('Window span', 'refit on sub-ranges of the temperature axis', '#ffffff', INK),
      ('Misfit-to-arc ratio  ρ = r$_{rms}$ / R$_{k,25}$', 'one division after the fit', '#fbeceb', RED)]
for i, (a, b, fc, ec) in enumerate(DG):
    y = 0.735 - i * 0.105
    rbox(x0, y, cw, 0.085, fc=fc, ec=ec, lw=0.9)
    t(x0 + cw / 2, y + 0.056, a, ha='center', va='center', fontsize=7.2, fontweight='bold', color=ec if i else '#555')
    t(x0 + cw / 2, y + 0.024, b, ha='center', va='center', fontsize=6.4)
down(x0 + cw / 2, 0.520, 0.470)
fig.patches.append(Polygon([(x0 + cw / 2, 0.462), (x0 + cw / 2 + 0.085, 0.405), (x0 + cw / 2, 0.348), (x0 + cw / 2 - 0.085, 0.405)], closed=True,
                           transform=fig.transFigure, fc='#fdf1e3', ec=ORG, lw=0.9, zorder=2))
t(x0 + cw / 2, 0.405, 'ρ ≤ 0.015 ?', ha='center', va='center', fontsize=7.4, fontweight='bold', color=ORG)
for side, lab, a, b, fc, ec in [(-1, 'yes', 'Trusted label', 'follow over aging,' + NL + 'use for compensation', '#eaf4ea', '#2e7d32'),
                                (1, 'no', 'Check the span', 'report the label' + NL + 'with its window span', '#fbeceb', RED)]:
    bx = x0 + (0.0 if side < 0 else cw / 2 + 0.006); bw = cw / 2 - 0.006
    rbox(bx, 0.135, bw, 0.13, fc=fc, ec=ec, lw=0.9)
    t(bx + bw / 2, 0.235, a, ha='center', va='center', fontsize=7.2, fontweight='bold', color=ec)
    t(bx + bw / 2, 0.180, b, ha='center', va='center', fontsize=6.0, linespacing=1.2)
    xs = x0 + cw / 2 + side * 0.085
    fig.patches.append(FancyArrowPatch((xs, 0.405), (xs, 0.268), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=8,
                                       shrinkA=0, shrinkB=0, color=INK, lw=0.9, zorder=3))
    t(xs + side * 0.006, 0.415, lab, ha='left' if side > 0 else 'right', va='bottom', fontsize=6.6)
fig.savefig(os.path.join(OUT, 'fig1_v11.png'), dpi=600, facecolor='white'); print('saved')
# ---- automatic check: every pair of visible texts (figure texts, axes texts, tick labels, axis labels) must not overlap
import sys; sys.stdout.reconfigure(encoding='utf-8')
fig.canvas.draw(); r = fig.canvas.get_renderer(); items = []
for tx in fig.texts: items.append(tx)
for a in fig.axes:
    if not a.axison: items += a.texts; continue
    def inview(axis, lim):
        lo, hi = sorted(lim); return [tl for loc, tl in zip(axis.get_majorticklocs(), axis.get_majorticklabels()) if lo <= loc <= hi]
    items += a.texts + [a.xaxis.label, a.yaxis.label, a.title] + inview(a.xaxis, a.get_xlim()) + inview(a.yaxis, a.get_ylim())
    for ch in a.get_children():
        if type(ch).__name__ == 'SecondaryAxis': items += inview(ch.xaxis, ch.get_xlim())
bb = [(tx.get_text(), tx.get_window_extent(r)) for tx in items if tx.get_visible() and tx.get_text().strip()]
bad = 0
for i in range(len(bb)):
    for j in range(i + 1, len(bb)):
        if bb[i][1].overlaps(bb[j][1]): bad += 1; print('OVERLAP:', repr(bb[i][0]), '<->', repr(bb[j][0]))
print('texts:', len(bb), '| overlapping pairs:', bad)
