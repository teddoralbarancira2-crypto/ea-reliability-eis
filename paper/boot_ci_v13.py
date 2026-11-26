# -*- coding: utf-8 -*-
"""Paper 4 v11: does the Jacobian 95 % CI of the dataset B slow-process Ea_R cover what the data support once the residual
structure is respected? Two bootstraps of the full 10-temperature K = 2 joint fit (same model and start as ci_dataset_b.py):
 (T) temperature bootstrap: resample the 10 spectra (whole temperatures) with replacement and refit -> percentile interval;
 (R) residual-block bootstrap: keep the design, add the fitted residual vectors of whole temperatures permuted across the
     temperatures (frequency structure kept, its position on the temperature axis randomized) to the fitted model, refit.
B = 1000 each (v13: endpoint stability over five blocks of 200), 4 cases. Writes boot_ci_v13.json. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import os, sys, json, numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
AL = config.AL; sys.path.insert(0, AL)
H = os.path.dirname(os.path.abspath(__file__))
kB = 8.617333e-5; TREF = 298.15; K = 2; NB = 1000

def resid(p, data, Xs):
    n = len(data); R0 = p[:n]; L = np.exp(p[n:2 * n]); q = p[2 * n:]; out = []
    for i, (f, zr, zi) in enumerate(data):
        w = 2 * np.pi * f; z = R0[i] + 1j * w * L[i]
        for k in range(K):
            lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]; z = z + np.exp(lR + Ea * Xs[i]) / (1 + (1j * w * np.exp(lt + Et * Xs[i])) ** a)
        lQ, Eq, nn = q[5 * K:5 * K + 3]; z = z + 1 / (np.exp(lQ - Eq * Xs[i]) * (1j * w) ** nn)
        out += [z.real - zr, -z.imag - zi]
    return np.concatenate(out)

def slow_ea(q):
    ks = sorted((0, 5), key=lambda b: q[b + 2]); return float(q[ks[1] + 1])

def setup(case):
    os.chdir(AL); from drt_all import load, TEMPS
    MS = json.load(open(os.path.join(AL, 'joint_ms_K2.json'))); cell, soc = case.rsplit('_', 1)
    data = [(f, zr * 1e3, zi * 1e3) for f, zr, zi in (load(cell, t, int(soc)) for t in TEMPS)]
    T = np.array(TEMPS, float); Xs = 1 / (kB * (T + 273.15)) - 1 / (kB * TREF); n = len(T)
    r = least_squares(resid, np.array(MS[case]['x']), args=(data, Xs), x_scale='jac', max_nfev=4000)
    return data, T, Xs, r.x, n

def job(a):
    case, kind, b = a; data, T, Xs, xh, n = setup(case); rng = np.random.default_rng(1000 * b + (7 if kind == 'R' else 3))
    if kind == 'T':
        while True:
            ix = rng.integers(0, n, n)
            if len(set(ix)) >= 4 and T[ix].max() - T[ix].min() >= 25: break
        d2 = [data[i] for i in ix]; X2 = Xs[ix]; x0 = np.r_[xh[:n][ix], xh[n:2 * n][ix], xh[2 * n:]]
    else:
        res = resid(xh, data, Xs); m = len(res) // n; blocks = [res[i * m:(i + 1) * m] for i in range(n)]; perm = rng.permutation(n)
        d2 = []
        for i, (f, zr, zi) in enumerate(data):
            fit_r = zr + blocks[i][:len(f)]; fit_i = zi + blocks[i][len(f):]                      # model values at T_i
            e = blocks[perm[i]]; d2.append((f, fit_r + e[:len(f)], fit_i + e[len(f):]))
        X2 = Xs; x0 = xh.copy()
    try:
        r = least_squares(resid, x0, args=(d2, X2), x_scale='jac', max_nfev=4000); return case, kind, slow_ea(r.x[2 * len(d2):])
    except Exception:
        return case, kind, None

if __name__ == '__main__':
    CASES = ('no1_ch3_50', 'no1_ch3_70', 'no2_ch4_50', 'no2_ch4_70')
    CI = json.load(open(os.path.join(H, 'ci_dataset_b.json')))
    with Pool(12) as pool: R = pool.map(job, [(c, k, b) for c in CASES for k in ('T', 'R') for b in range(NB)])
    out = {}
    for c in CASES:
        e0, se = CI[c]['slow']['Ea'], CI[c]['slow']['se']; row = dict(Ea=round(e0, 3), jac95=[round(e0 - 1.96 * se, 3), round(e0 + 1.96 * se, 3)])
        for k in ('T', 'R'):
            v = np.array([x for cc, kk, x in R if cc == c and kk == k and x is not None])
            blk = [v[i * 200:(i + 1) * 200] for i in range(len(v) // 200)]
            row[k] = dict(n=int(len(v)), p95=[round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)],
                          sd=round(float(v.std()), 3), width_ratio=round(float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / (3.92 * se)), 2),
                          blocks_lo=[round(float(np.percentile(b, 2.5)), 3) for b in blk], blocks_hi=[round(float(np.percentile(b, 97.5)), 3) for b in blk])
        out[c] = row; print(c, row, flush=True)
    json.dump(out, open(os.path.join(H, 'boot_ci_v13.json'), 'w'), indent=1); print('saved')
