# -*- coding: utf-8 -*-
"""Independence checks for the slow-process label (SOC step S):
 mode 'cold' : every checkpoint fitted on its own, default seed + 5 random seeds, no warm start (K=3 windows)
 mode 'back' : warm-start chain run backward, cycle 480 -> 40 (K=3 windows)
 mode 'k2'   : fast band as ONE ZARC (K=2: fast band + slow) + CPE, cold multistart; also reports the tail (Q25, n, Eq)
Every mode lists parameters within 1 % of a bound. Writes checks_{mode}_s{S}.json. py -3.11 checks.py MODE S"""
import numpy as np, json, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
from common import load, ok, TS, CYC, JS
import joint_age as JA
MODE = sys.argv[1]; S = int(sys.argv[2])
if MODE == 'k2':
    JA.K = 2; JA.TAUB = [(1e-5, 1.5e-2), (1.5e-2, 2.0)]; JA.SEED = [(5.0, 0.40, 8e-4, 0.30, 0.6), (2.0, 0.55, 5e-2, 0.55, 0.9)]
NAMES = ['lnR', 'EaR', 'lntau', 'Etau', 'alpha']

def multistart(data, xs, rng, n_rand=5, warm=None):
    n = len(data); lo, hi = JA.bounds(); head = list(np.array([d[1][0] for d in data]) * 0.98) + [np.log(1e-5)] * n
    lo = np.array([0] * n + [np.log(1e-8)] * n + lo); hi = np.array([100] * n + [np.log(1e-3)] * n + hi)
    starts = [head + sum([[np.log(R), Ea, np.log(t), Et, a] for R, Ea, t, Et, a in JA.SEED], []) + [np.log(20.0), 0.4, 0.6]]
    for _ in range(n_rand):
        p = list(head)
        for tl, th in JA.TAUB: p += [np.log(rng.uniform(0.3, 8)), rng.uniform(0.2, 0.8), rng.uniform(np.log(tl), np.log(th)), rng.uniform(0.2, 0.8), rng.uniform(0.5, 0.95)]
        p += [np.log(rng.uniform(5, 100)), rng.uniform(0.1, 0.7), rng.uniform(0.4, 0.8)]; starts.append(p)
    if warm is not None: starts = [head + list(warm)]
    best = None
    for p0 in starts:
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        r = least_squares(JA.resid, p0, args=(data, xs), bounds=(lo, hi), x_scale='jac', max_nfev=8000)
        if best is None or r.cost < best.cost: best = r
    # bound audit on the process + tail block
    q = best.x[2 * n:]; ql = lo[2 * n:]; qh = hi[2 * n:]; flags = []
    for i, (v, a, b) in enumerate(zip(q, ql, qh)):
        if (v - a) <= 0.01 * (b - a) or (b - v) <= 0.01 * (b - a):
            k = i // 5; flags.append('tail.%s' % ['lnQ', 'Eq', 'n'][i - 5 * JA.K] if k >= JA.K else 'p%d.%s' % (k, NAMES[i % 5]))
    return best, flags

def rec_of(r, n, flags, cyc):
    P, q = JA.procs(r, n); lQ, Eq, nn = q[5 * JA.K:5 * JA.K + 3]
    return dict(cyc=int(cyc), rms=float(np.sqrt(np.mean(r.fun ** 2))), proc=P, tail=dict(Q25=float(np.exp(lQ)), Eq=float(Eq), n=float(nn)), flags=flags, q=list(map(float, q)))

def prep(c, j):
    ks = [k for k in range(7) if ok(c, j, k)]; data = [load(c, j, k, S) for k in ks]
    return data, 1 / (JA.kB * (TS[ks] + 273.15)) - 1 / (JA.kB * JA.TREF)

def job_cold(a):
    c, j = a; data, xs = prep(c, j); r, fl = multistart(data, xs, np.random.default_rng(1000 * c + j)); return (c, j), rec_of(r, len(data), fl, CYC[j])

def job_back(c):
    out = []; warm = None
    for j in JS[::-1]:
        data, xs = prep(c, j)
        r, fl = multistart(data, xs, np.random.default_rng(j), n_rand=0) if warm is None else multistart(data, xs, None, n_rand=0, warm=warm)
        if warm is not None:                                  # also try the default seed, keep the better
            r2, fl2 = multistart(data, xs, np.random.default_rng(j), n_rand=0)
            if r2.cost < r.cost: r, fl = r2, fl2
        R = rec_of(r, len(data), fl, CYC[j]); warm = R['q']; out.append(((c, j), R))
    return out

if __name__ == '__main__':
    with Pool(8) as pool:
        if MODE == 'back': res = sum(pool.map(job_back, (0, 1)), [])
        else: res = pool.map(job_cold, [(c, j) for c in (0, 1) for j in JS])
    js = {'cell%d_%d' % (c + 1, CYC[j]): v for (c, j), v in res}
    json.dump(js, open('checks_%s_s%d.json' % (MODE, S), 'w'), indent=1)
    for key in sorted(js, key=lambda k: (k[:5], int(k.split('_')[1]))):
        v = js[key]; sl = v['proc'][-1]; fb = sum(p['R25'] for p in v['proc'][:-1])
        print('%-10s rms %.3f | slow R %.2f t %.3f EaR %.2f Eat %.2f a %.2f | fast-band R %.2f | tail Q %.1f n %.2f Eq %.2f | flags %s' % (
            key, v['rms'], sl['R25'], sl['tau25'], sl['Ea'], sl['Et'], sl['a'], fb, v['tail']['Q25'], v['tail']['n'], v['tail']['Eq'], ','.join(v['flags'])), flush=True)
