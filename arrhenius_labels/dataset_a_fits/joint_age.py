# -*- coding: utf-8 -*-
"""Arrhenius-coupled joint fit across the 7 temperatures at every aging checkpoint, per cell and SOC step.
K = 3 ZARCs (fast, mid, slow; each confined to its own tau25 window) + CPE tail; R_k(T), tau_k(T) Arrhenius about 25 C; R0, L free per T. f <= 2 kHz.
Each fit tries the default seed plus a warm start from the previous checkpoint and keeps the best.
Optional drop-one-temperature jackknife (JK=1). Writes joint_age_s{S}.json. py -3.11 joint_age.py S [JK]"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
from common import load, ok, TS, CYC, JS
kB = 8.617333e-5; TREF = 298.15; K = 3
SEED = [(1.0, 0.40, 3e-4, 0.40, 0.7), (2.0, 0.40, 3e-3, 0.50, 0.8), (2.0, 0.55, 5e-2, 0.55, 0.9)]
TAUB = [(1e-5, 1.5e-3), (1.5e-3, 1.5e-2), (1.5e-2, 2.0)]        # tau25 window per process: fast, mid, slow (no label switching)

def resid(p, data, xs):
    n = len(data); R0 = p[:n]; L = np.exp(p[n:2 * n]); q = p[2 * n:]; out = []
    for i, (f, zr, zi) in enumerate(data):
        w = 2 * np.pi * f; z = R0[i] + 1j * w * L[i]
        for k in range(K):
            lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]
            z = z + np.exp(lR + Ea * xs[i]) / (1 + (1j * w * np.exp(lt + Et * xs[i])) ** a)
        lQ, Eq, nn = q[5 * K:5 * K + 3]; z = z + 1 / (np.exp(lQ - Eq * xs[i]) * (1j * w) ** nn)
        out += [z.real - zr, -z.imag - zi]
    return np.concatenate(out)

def bounds():
    lo, hi = [], []
    for tl, th in TAUB: lo += [np.log(1e-3), -0.5, np.log(tl), -0.5, 0.4]; hi += [np.log(500), 2.0, np.log(th), 2.0, 1.0]
    lo += [np.log(0.01), -0.5, 0.3]; hi += [np.log(1e5), 2.0, 1.0]; return lo, hi

def fit(data, xs, warm=None):
    n = len(data); lo, hi = bounds()
    head = list(np.array([d[1][0] for d in data]) * 0.98) + [np.log(1e-5)] * n
    lo = [0] * n + [np.log(1e-8)] * n + lo; hi = [100] * n + [np.log(1e-3)] * n + hi
    starts = [head + sum([[np.log(R), Ea, np.log(t), Et, a] for R, Ea, t, Et, a in SEED], []) + [np.log(20.0), 0.4, 0.6]]
    if warm is not None: starts.append(head + list(warm))
    best = None
    for p0 in starts:
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        r = least_squares(resid, p0, args=(data, xs), bounds=(lo, hi), x_scale='jac', max_nfev=8000)
        if best is None or r.cost < best.cost: best = r
    return best

def procs(r, n):
    q = r.x[2 * n:]; P = []
    for k in range(K):
        lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]; P.append(dict(R25=float(np.exp(lR)), Ea=float(Ea), tau25=float(np.exp(lt)), Et=float(Et), a=float(a)))
    order = np.argsort([p['tau25'] for p in P]); return [P[i] for i in order], q

def run_cell(args):
    c, s, jk = args; out = []; warm = None
    for j in JS:
        ks = [k for k in range(7) if ok(c, j, k)]; data = [load(c, j, k, s) for k in ks]
        xs = 1 / (kB * (TS[ks] + 273.15)) - 1 / (kB * TREF)
        r = fit(data, xs, warm); P, q = procs(r, len(ks)); warm = q
        rec = dict(cyc=int(CYC[j]), temps=[int(TS[k]) for k in ks], rms=float(np.sqrt(np.mean(r.fun ** 2))), proc=P, q=list(map(float, q)))
        if jk:
            rec['jk'] = []
            for d in range(len(ks)):
                keep = [i for i in range(len(ks)) if i != d]
                rj = fit([data[i] for i in keep], xs[keep], q); rec['jk'].append(dict(drop=int(TS[ks[d]]), proc=procs(rj, len(keep))[0]))
        out.append(rec)
        print('cell%d s%d cyc%3d rms %.3f | ' % (c + 1, s, CYC[j], rec['rms']) + ' | '.join(
            'R %.2f t %.2g EaR %.2f Eat %.2f a %.2f' % (p['R25'], p['tau25'], p['Ea'], p['Et'], p['a']) for p in P), flush=True)
    return (c, s), out

if __name__ == '__main__':
    S = int(sys.argv[1]); JK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with Pool(2) as pool: res = dict(pool.map(run_cell, [(c, S, JK) for c in (0, 1)]))
    json.dump({'cell%d' % (c + 1): v for (c, s), v in res.items()}, open('joint_age_s%d%s.json' % (S, '_jk' if JK else ''), 'w'), indent=1)
