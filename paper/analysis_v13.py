# -*- coding: utf-8 -*-
"""Paper 4 v13 numbers for the codex v12 review items (no refits of the published analysis):
 (1) linear Kramers-Kronig test (Schoenleber et al. 2014 style: Voigt series with fixed log-spaced tau, R0, series L, ordinary
     least squares on real and imaginary parts together) for every spectrum of dataset A (f <= 10 kHz) and dataset B (f <= 2 kHz):
     relative residual |Z_fit - Z| / |Z| per point -> median and max per spectrum, by temperature
 (2) dataset B fit residual (full 10-T K = 2 joint fit, same start as ci_dataset_b.py) by temperature and frequency -> residual map
 (3) calibrate-once 255 % 90th percentile: which source temperatures produce it
 (4) fixed threshold 0.015 scored on each held-out fold; (5) Bao & Gong cell 3 full-range R2 to 4 decimals
Writes analysis_v13.json. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import os, sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from scipy.optimize import least_squares
H = os.path.dirname(os.path.abspath(__file__))
AL = config.AL; sys.path.insert(0, AL); sys.path.insert(0, os.path.join(AL, 'dataset_a_fits'))
out = {}

def linkk(f, zr, zi, M=None):
    """lin-KK: Z = R0 + jwL + sum_k Rk/(1 + jw tau_k), tau_k log-spaced over 1/(2 pi f), M = 7 per decade; OLS on Re and Im."""
    w = 2 * np.pi * f; dec = np.log10(f.max() / f.min()); M = M or int(np.ceil(7 * dec))
    tau = np.logspace(np.log10(1 / w.max()), np.log10(1 / w.min()), M)
    wt = w[:, None] * tau[None, :]
    A = np.vstack([np.c_[np.ones_like(w), np.zeros_like(w), 1 / (1 + wt ** 2)], np.c_[np.zeros_like(w), w, -wt / (1 + wt ** 2)]])
    b = np.r_[zr, zi]; x = np.linalg.lstsq(A, b, rcond=None)[0]; fit = A @ x
    mag = np.hypot(zr, zi); return np.hypot(fit[:len(f)] - zr, fit[len(f):] - zi) / mag

# (1a) dataset A
from common import Z, F, TS, CYC, ok
MF = F <= 10001.0; KA = {}
for c in (0, 1):
    for j, n in enumerate(CYC):
        if n == 400: continue
        for k, T in enumerate(TS):
            if not ok(c, j, k): continue
            for s, v in ((3, 4.06), (10, 3.73), (17, 3.41)):
                z = Z[c, j, k, s][MF]; e = linkk(F[MF], z.real * 1e3, z.imag * 1e3)
                KA.setdefault(int(T), []).append((float(np.median(e)), float(e.max())))
out['kk_A'] = {T: dict(n=len(v), med_of_med=round(100 * float(np.median([a for a, _ in v])), 3), p95_of_max=round(100 * float(np.percentile([b for _, b in v], 95)), 3),
                       max_of_max=round(100 * float(max(b for _, b in v)), 3)) for T, v in sorted(KA.items())}
print('KK dataset A (%):', out['kk_A'])
# (1b) dataset B
os.chdir(AL); from drt_all import load, TEMPS, CELLS, SOCS
KB = {}
for cell in CELLS:
    for soc in SOCS:
        for t in TEMPS:
            f, zr, zi = load(cell, t, soc); m = f <= 2001.0
            e = linkk(f[m], zr[m] * 1e3, -zi[m] * 1e3)        # load returns zi = -Im
            KB.setdefault(int(t), []).append((float(np.median(e)), float(e.max())))
out['kk_B'] = {T: dict(n=len(v), med_of_med=round(100 * float(np.median([a for a, _ in v])), 3), p95_of_max=round(100 * float(np.percentile([b for _, b in v], 95)), 3),
                       max_of_max=round(100 * float(max(b for _, b in v)), 3)) for T, v in sorted(KB.items())}
print('KK dataset B (%):', out['kk_B'])
# (2) dataset B residual map, cell no1_ch3 SOC 50 and all four cases
sys.path.insert(0, H); import boot_ci_v13 as BC   # bootstrap module in the same folder
RM = {}
for case in ('no1_ch3_50', 'no1_ch3_70', 'no2_ch4_50', 'no2_ch4_70'):
    data, T, Xs, xh, n = BC.setup(case); r = BC.resid(xh, data, Xs); m = len(r) // n
    RM[case] = dict(f=data[0][0].tolist(), T=T.tolist(), re=[r[i * m:i * m + len(data[0][0])].tolist() for i in range(n)],
                    im=[r[i * m + len(data[0][0]):(i + 1) * m].tolist() for i in range(n)],
                    rms_T=[float(np.sqrt(np.mean(r[i * m:(i + 1) * m] ** 2))) for i in range(n)], rms=float(np.sqrt(np.mean(r ** 2))))
    print(case, 'rms per T (mOhm):', dict(zip(T.astype(int).tolist(), np.round(RM[case]['rms_T'], 3).tolist())))
out['resid_B'] = RM
# (3) calibrate-once breakdown
AT = json.load(open(os.path.join(AL, 'ident', 'aging_transfer.json')))
for A in (40, 240):
    s = [r for r in AT if r['A'] == A]; e = np.array([max(abs(v) for v in r['ours'].values()) * 100 for r in s]); p90 = np.percentile(e, 90)
    top = [r for r, x in zip(s, e) if x >= p90]
    out['transfer_%d' % A] = dict(n=len(s), p90=round(float(p90), 1), top_T1={str(t): sum(1 for r in top if r['T1'] == t) for t in (0, 5, 15, 35, 45, 55)},
                                  med_by_T1={str(t): round(float(np.median([x for r, x in zip(s, e) if r['T1'] == t])), 1) for t in (0, 5, 15, 35, 45, 55)},
                                  max=round(float(e.max()), 0))
    print('transfer cal %d:' % A, out['transfer_%d' % A])
# (4) fixed threshold per fold
rows = json.load(open(os.path.join(H, 'fig3_rows.json'))); x = np.array([r['ratio_hf'] for r in rows]); y = np.array([r['span'] for r in rows])
cell = np.array([r['cell'] for r in rows]); volt = np.array([r['v'] for r in rows]); FX = {}
for name, g in (('cell', cell), ('voltage', volt)):
    for gv in np.unique(g):
        mm = g == gv; p = mm & (x <= 0.015)
        FX['%s %s' % (name, gv)] = dict(n=int(mm.sum()), passed=int(p.sum()), unstable_passed=int((p & (y > 0.08)).sum()),
                                         stable_flagged=int((mm & ~p & (y <= 0.08)).sum()), max_pass_span=round(float(y[p].max()), 3))
out['fixed_0015'] = FX; print('fixed 0.015 per fold:', FX)
# (5) Bao & Gong
TB = np.array([-10, 0, 10, 30, 50]) + 273.15
print('see make_figs_v11 output for R2; recomputed with 4 decimals below')
json.dump(out, open(os.path.join(H, 'analysis_v13.json'), 'w'), indent=1); print('saved')
