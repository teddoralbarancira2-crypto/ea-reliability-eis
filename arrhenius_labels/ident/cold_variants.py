# -*- coding: utf-8 -*-
"""Follow-up to cold_split.py: which misspecification drives the dataset B slow-Ea swing? K=2 joint fit variants:
 base   as before (CPE tail Q(T) Arrhenius, n shared; arc widths fixed)
 tailT  CPE tail Q and n FREE per temperature (diffusion end not forced to scale)
 alphaT fast-arc width alpha_fast(T) = a0 + a1 X
 both   tailT + alphaT
Per variant: per-T rms, BIC, slow Ea_R all / drop 0 C / drop 5 C. Writes cold_variants.json. py -3.11"""
import numpy as np, json, os
from multiprocessing import Pool
from scipy.optimize import least_squares
from cold_split import getdata, XS, TT, CELLS

def unpack(q, n, v):
    R0 = q[:n]; L = np.exp(q[n:2 * n]); p = q[2 * n:2 * n + 10]; r = q[2 * n + 10:]
    if v in ('tailT', 'both'): lQ, nn, r = r[:n], r[n:2 * n], r[2 * n:]
    else: lQ, nn, r = None, None, r
    if v in ('base', 'alphaT'): tail = r[:3]; r = r[3:]
    else: tail = None
    a1 = r[0] if v in ('alphaT', 'both') else 0.0
    return R0, L, p, lQ, nn, tail, a1

def model(q, xs, w, v):
    n = len(xs); R0, L, p, lQ, nn, tail, a1 = unpack(q, n, v); out = []
    for i, x in enumerate(xs):
        z = R0[i] + 1j * w * L[i]
        for k in range(2):
            lR, Ea, lt, Et, a = p[5 * k:5 * k + 5]
            if k == 0: a = np.clip(a + a1 * x, 0.3, 1.0)
            z = z + np.exp(lR + Ea * x) / (1 + (1j * w * np.exp(lt + Et * x)) ** a)
        if tail is not None: Q, m = np.exp(tail[0] - tail[1] * x), tail[2]
        else: Q, m = np.exp(lQ[i]), nn[i]
        out.append(z + 1 / (Q * (1j * w) ** m))
    return out

def resid(q, xs, w, data, v):
    return np.concatenate([np.concatenate([z.real - d.real, z.imag - d.imag]) for z, d in zip(model(q, xs, w, v), data)])

def fit(data, xs, w, v, seed=7, nstart=6):
    n = len(xs); rng = np.random.default_rng(seed); r0 = [d.real[np.argmin(np.abs(w / 2 / np.pi - 1000))] * 0.98 for d in data]
    lo = [0] * n + [np.log(1e-6)] * n + [np.log(1e-3), -0.5, np.log(1e-5), -0.5, 0.4] * 2; hi = [50] * n + [np.log(5e-3)] * n + [np.log(100), 1.5, np.log(0.1), 1.5, 1.0] * 2
    if v in ('tailT', 'both'): lo += [np.log(0.01)] * n + [0.3] * n; hi += [np.log(1e5)] * n + [1.0] * n
    else: lo += [np.log(0.01), -0.5, 0.3]; hi += [np.log(1e5), 1.5, 1.0]
    if v in ('alphaT', 'both'): lo += [-0.1]; hi += [0.1]
    lo, hi = np.array(lo), np.array(hi); best = None
    for s in range(nstart):
        p = list(r0) + [np.log(5e-4)] * n
        for t in (4e-4, 3e-3):
            p += [np.log(2.0 if s == 0 else rng.uniform(0.3, 3)), 0.45 if s == 0 else rng.uniform(0.2, 0.8), np.log(t * (1 if s == 0 else np.exp(rng.uniform(-1, 1)))), 0.5 if s == 0 else rng.uniform(0.2, 0.8), 0.8 if s == 0 else rng.uniform(0.5, 0.95)]
        p += ([np.log(0.8)] * n + [0.6] * n) if v in ('tailT', 'both') else [np.log(0.8), 0.05, 0.6]
        if v in ('alphaT', 'both'): p += [0.0]
        p = np.clip(p, lo + 1e-6, hi - 1e-6)
        try: r = least_squares(resid, p, args=(xs, w, data, v), bounds=(lo, hi), x_scale='jac', max_nfev=8000)
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    q = best.x; _, _, p, _, nn, _, a1 = unpack(q, n, v); nf = len(w)
    perT = [float(np.sqrt(np.mean(best.fun[2 * i * nf:2 * (i + 1) * nf] ** 2))) for i in range(n)]
    N = len(best.fun); k = int(np.argmax([p[2], p[7]]))
    return dict(slowEaR=float(p[5 * k + 1]), slowR=float(np.exp(p[5 * k])), fastEaR=float(p[5 * (1 - k) + 1]), perT=perT, rms=float(np.sqrt(np.mean(best.fun ** 2))),
                bic=float(N * np.log(np.mean(best.fun ** 2)) + len(q) * np.log(N)), a1=float(a1), n_tail=None if nn is None else [float(x) for x in nn])

def job(a):
    cell, soc, v = a; w, data = getdata(cell, soc); out = {}
    for tag, keep in (('all', TT >= 0), ('drop0', TT != 0), ('drop5', TT != 5)):
        out[tag] = fit([d for d, k in zip(data, keep) if k], XS[keep], w, v)
    return '%s_%d_%s' % a, out

if __name__ == '__main__':
    jobs = [(c, s, v) for c in CELLS for s in (50, 70) for v in ('base', 'tailT', 'alphaT', 'both')]
    with Pool(16) as pool: R = dict(pool.map(job, jobs))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cold_variants.json'), 'w'), indent=1)
    for k, o in R.items():
        a = o['all']; s = [o[t]['slowEaR'] for t in ('all', 'drop0', 'drop5')]
        print('%-20s rms %.4f BIC %6.0f | perT 0/5/25/45 %s | slow EaR %.3f d0 %.3f d5 %.3f swing %.3f | fast EaR %.3f | a1 %.3f%s' % (
            k, a['rms'], a['bic'], '/'.join('%.3f' % a['perT'][i] for i in (0, 1, 5, 9)), s[0], s[1], s[2], max(abs(s[1] - s[0]), abs(s[2] - s[0])), a['fastEaR'], a['a1'],
            '' if a['n_tail'] is None else ' | n_tail 0/5/25/45 ' + '/'.join('%.2f' % a['n_tail'][i] for i in (0, 1, 5, 9))))
