# -*- coding: utf-8 -*-
"""Multistart version of the joint Arrhenius-coupled fit (joint_fit.py), K processes, N random starts per (cell, SOC),
keeps the lowest-cost fit. Also reports the SE of the Ea difference between the two largest processes (full covariance).
Writes joint_ms_K{K}.json. py -3.11 joint_ms.py K N"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
from drt_all import load, CELLS, TEMPS, SOCS
from joint_fit import resid_joint, X
HERE = os.path.dirname(os.path.abspath(__file__))

def one(args):
    cell, soc, K, N = args
    data = [(f, zr * 1e3, zi * 1e3) for f, zr, zi in (load(cell, t, soc) for t in TEMPS)]; nT = len(data)
    rng = np.random.default_rng(hash((cell, soc, K)) % 2**32); best = None
    base_lo = [0] * nT + [np.log(1e-5)] * nT; base_hi = [50] * nT + [np.log(5e-3)] * nT
    for i in range(N):
        taus = np.sort(10 ** rng.uniform(-4, 0, K)) if i else np.array({2: [1e-3, 2e-2], 3: [5e-4, 5e-3, 5e-2], 4: [3e-4, 3e-3, 3e-2, 0.3]}[K])
        p0 = list(np.array([d[1][np.argmin(np.abs(d[0] - 1000))] for d in data]) * 0.98) + [np.log(5e-4)] * nT
        lo = list(base_lo); hi = list(base_hi)
        for t in taus:
            p0 += [np.log(rng.uniform(0.5, 5)), rng.uniform(0.2, 0.7), np.log(t), rng.uniform(0.2, 0.7), rng.uniform(0.6, 0.95)]
            lo += [np.log(1e-3), -0.5, np.log(1e-6), -0.5, 0.4]; hi += [np.log(100), 1.5, np.log(50), 1.5, 1.0]
        p0 += [np.log(50.0), 0.3, 0.6]; lo += [np.log(0.1), -0.5, 0.3]; hi += [np.log(1e5), 1.5, 1.0]
        try: r = least_squares(resid_joint, p0, args=(data, K), bounds=(lo, hi), x_scale='jac', max_nfev=6000)
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    r = best; J = r.jac; dof = len(r.fun) - len(r.x); s2 = (r.fun ** 2).sum() / dof
    cov = np.linalg.pinv(J.T @ J) * s2; q0 = 2 * nT; procs = []
    for k in range(K):
        b = q0 + 5 * k; lR, Ea, lt, Et, a = r.x[b:b + 5]
        procs.append(dict(i=b, R25=float(np.exp(lR)), Ea=float(Ea), Ea_se=float(np.sqrt(cov[b + 1, b + 1])), tau25=float(np.exp(lt)),
                          Et=float(Et), Et_se=float(np.sqrt(cov[b + 3, b + 3])), a=float(a)))
    procs.sort(key=lambda d: d['tau25'])
    big = sorted(procs, key=lambda d: -d['R25'])[:2]; big.sort(key=lambda d: d['tau25'])
    def dse(j):  # SE of Ea(fast) - Ea(slow) among the two largest processes, j = 1 (R) or 3 (tau)
        i1, i2 = big[0]['i'] + j, big[1]['i'] + j; return float(np.sqrt(cov[i1, i1] + cov[i2, i2] - 2 * cov[i1, i2]))
    out = dict(rms=float(np.sqrt(np.mean(r.fun ** 2))), proc=procs, dEa=big[1]['Ea'] - big[0]['Ea'], dEa_se=dse(1),
               dEt=big[1]['Et'] - big[0]['Et'], dEt_se=dse(3), R0=list(map(float, r.x[:nT])), x=list(map(float, r.x)))
    return '%s_%d' % (cell, soc), out

if __name__ == '__main__':
    K = int(sys.argv[1]); N = int(sys.argv[2])
    with Pool(6) as pool: res = dict(pool.map(one, [(c, s, K, N) for c in CELLS for s in SOCS]))
    for k, v in res.items():
        print('%-12s rms %.4f | ' % (k, v['rms']) + ' | '.join('R %.2f tau %.2g EaR %.3f±%.3f Eat %.3f±%.3f a %.2f' % (
            d['R25'], d['tau25'], d['Ea'], d['Ea_se'], d['Et'], d['Et_se'], d['a']) for d in v['proc']) +
              '  || dEaR %.3f±%.3f dEat %.3f±%.3f' % (v['dEa'], v['dEa_se'], v['dEt'], v['dEt_se']))
    json.dump(res, open(os.path.join(HERE, 'joint_ms_K%d.json' % K), 'w'), indent=1)
