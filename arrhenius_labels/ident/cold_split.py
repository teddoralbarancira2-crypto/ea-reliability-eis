# -*- coding: utf-8 -*-
"""Cold-peak splitting test (hypothesis from Heins et al. 2020: processes that overlap at room T separate below ~10 C
because their Ea differ; a fixed-K model then misfits the cold spectra and that misfit drives the dataset B slow-Ea swing).
dataset B, 2 cells x SOC 50/70, 10 T (0-45 C), f <= 2 kHz.
 A  model-free: DRT peak count/positions per T at three lambdas (same tau grid as drt_all.py).
 B  where the K=2 joint-fit misfit sits in frequency at each T.
 C  decisive: K=3 Arrhenius-coupled joint fit. Does the cold misfit vanish (per-T rms), does BIC prefer it, does the
    third process carry its own consistent Ea, and does the drop-0 C / drop-5 C swing of the slow-process Ea shrink?
    Control: K=3 on a well-specified synthetic K=2 set must not invent a sizeable third process.
Writes cold_split.json. py -3.11"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from drt_all import load, drt, TEMPS, TAU, CELLS
kB = 8.617333e-5; TREF = 298.15; TT = np.array(TEMPS, float); XS = 1 / (kB * (TT + 273.15)) - 1 / (kB * TREF)

def model(q, xs, w, K):
    n = len(xs); R0 = q[:n]; L = np.exp(q[n:2 * n]); p = q[2 * n:]; out = []
    for i, x in enumerate(xs):
        z = R0[i] + 1j * w * L[i]
        for k in range(K):
            lR, Ea, lt, Et, a = p[5 * k:5 * k + 5]; z = z + np.exp(lR + Ea * x) / (1 + (1j * w * np.exp(lt + Et * x)) ** a)
        lQ, Eq, nn = p[5 * K:5 * K + 3]; out.append(z + 1 / (np.exp(lQ - Eq * x) * (1j * w) ** nn))
    return out

def resid(q, xs, w, data, K):
    return np.concatenate([np.concatenate([z.real - d.real, z.imag - d.imag]) for z, d in zip(model(q, xs, w, K), data)])

def fit(data, xs, w, K, seed=0, nstart=8):
    n = len(xs); rng = np.random.default_rng(seed)
    r0 = [d.real[np.argmin(np.abs(w / 2 / np.pi - 1000))] * 0.98 for d in data]
    lo = [0] * n + [np.log(1e-6)] * n; hi = [50] * n + [np.log(5e-3)] * n
    for _ in range(K): lo += [np.log(1e-3), -0.5, np.log(1e-5), -0.5, 0.4]; hi += [np.log(100), 1.5, np.log(0.1), 1.5, 1.0]
    lo += [np.log(0.01), -0.5, 0.3]; hi += [np.log(1e5), 1.5, 1.0]; lo, hi = np.array(lo), np.array(hi)
    base_tau = {2: [4e-4, 3e-3], 3: [3e-4, 1.2e-3, 5e-3]}[K]; best = None
    for s in range(nstart):
        p = list(r0) + [np.log(5e-4)] * n
        for k in range(K):
            t = base_tau[k] * (1 if s == 0 else np.exp(rng.uniform(-1, 1)))
            p += [np.log(2.0 if s == 0 else rng.uniform(0.3, 3)), 0.45 if s == 0 else rng.uniform(0.2, 0.8), np.log(t), 0.5 if s == 0 else rng.uniform(0.2, 0.8), 0.8 if s == 0 else rng.uniform(0.5, 0.95)]
        p += [np.log(0.8), 0.05, 0.6]; p = np.clip(p, lo + 1e-6, hi - 1e-6)
        try: r = least_squares(resid, p, args=(xs, w, data, K), bounds=(lo, hi), x_scale='jac', max_nfev=8000)
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    q = best.x; p = q[2 * n:]; procs = sorted([dict(R25=float(np.exp(p[5 * k])), EaR=float(p[5 * k + 1]), tau25=float(np.exp(p[5 * k + 2])), Eat=float(p[5 * k + 3]), a=float(p[5 * k + 4])) for k in range(K)], key=lambda d: d['tau25'])
    nf = len(w); perT = [float(np.sqrt(np.mean(best.fun[2 * i * nf:2 * (i + 1) * nf] ** 2))) for i in range(n)]
    N = len(best.fun); bic = N * np.log(np.mean(best.fun ** 2)) + len(q) * np.log(N)
    return dict(procs=procs, perT=perT, rms=float(np.sqrt(np.mean(best.fun ** 2))), bic=float(bic), fun=best.fun.tolist())

def getdata(cell, soc):
    D = [load(cell, t, soc) for t in TEMPS]; f = D[0][0]
    return 2 * np.pi * f, [(zr - 1j * zi) * 1e3 for _, zr, zi in D]     # complex Z in mOhm

def slow_ea(res, K):
    """Ea_R of the process nearest the K=2 slow process (tau25 ~3 ms) and of the largest-tau arc below 20 ms."""
    c = [p for p in res['procs'] if 1e-3 < p['tau25'] < 2e-2]
    return max(c, key=lambda p: p['R25'])['EaR'] if c else np.nan

def jobC(args):
    cell, soc, K = args; w, data = getdata(cell, soc); out = {}
    for tag, keep in (('all', TT >= 0), ('drop0', TT != 0), ('drop5', TT != 5)):
        r = fit([d for d, k in zip(data, keep) if k], XS[keep], w, K, seed=7)
        if tag != 'all': r.pop('fun')
        out[tag] = r
    return (cell, soc, K), out

def jobA():
    out = {}
    for cell in CELLS:
        for soc in (50, 70):
            for lam in (0.005, 0.05, 0.5):
                rows = []
                for t in TEMPS:
                    f, zr, zi = load(cell, t, soc); _, _, g, _ = drt(f, zr, zi, lam)
                    m = (TAU > 3e-5) & (TAU < 5e-2); gg = g[m]; pk, pr = find_peaks(gg, prominence=0.05 * gg.max())
                    rows.append(dict(T=t, n=len(pk), tau=[float(x) for x in TAU[m][pk]], h=[float(x) for x in gg[pk]]))
                out['%s_%d_lam%g' % (cell, soc, lam)] = rows
    return out

if __name__ == '__main__':
    A = jobA()
    print('== A: DRT peaks between 30 us and 50 ms (prominence > 5 % of max)')
    for k, rows in A.items():
        print(k, ' '.join('%dC:%d[%s]' % (r['T'], r['n'], ','.join('%.2g' % t for t in r['tau'])) for r in rows))
    jobs = [(c, s, K) for c in CELLS for s in (50, 70) for K in (2, 3)]
    with Pool(8) as pool: C = dict(pool.map(jobC, jobs))
    print('== B: K=2 misfit by frequency band at each T (rms mOhm), cell no1_ch3 SOC 50')
    w, data = getdata('no1_ch3', 50); f = w / 2 / np.pi; nf = len(f); fun = np.array(C[('no1_ch3', 50, 2)]['all']['fun'])
    bands = [(500, 2001), (50, 500), (5, 50), (0.5, 5), (0.09, 0.5)]
    for i, t in enumerate(TEMPS):
        e = fun[2 * i * nf:2 * (i + 1) * nf]; e = np.hypot(e[:nf], e[nf:])
        print('%2dC ' % t + ' '.join('%g-%gHz %.3f' % (a, b, np.sqrt(np.mean(e[(f >= a) & (f < b)] ** 2))) for a, b in bands))
    print('== C: K=2 vs K=3 joint fits')
    for (cell, soc, K), o in sorted(C.items()):
        a = o['all']; s = [slow_ea(o[t], K) for t in ('all', 'drop0', 'drop5')]
        print('%s SOC%d K=%d rms %.4f BIC %.0f | perT 0/5/25/45C %s | slow EaR all %.3f drop0 %.3f drop5 %.3f swing %.3f' % (
            cell, soc, K, a['rms'], a['bic'], '/'.join('%.3f' % a['perT'][i] for i in (0, 1, 5, 9)), s[0], s[1], s[2], np.nanmax(np.abs(np.array(s[1:]) - s[0]))))
        print('     ' + ' | '.join('R25 %.2f tau %.2g EaR %.2f Eat %.2f a %.2f' % (p['R25'], p['tau25'], p['EaR'], p['Eat'], p['a']) for p in a['procs']))
    # control: synthetic well-specified K=2 (dataset B geometry, noise 0.015) fitted with K=3
    import synth_ident as S
    wS = 2 * np.pi * S.F; q = S.truth(7, 0.3); rng = np.random.default_rng(3)
    syn = [z + rng.normal(0, 0.015, len(wS)) + 1j * rng.normal(0, 0.015, len(wS)) for z in S.model(q, S.XS, wS)]
    ctl = fit(syn, S.XS, wS, 3, seed=7); ctl.pop('fun')
    print('== control K=3 on synthetic K=2: rms %.4f | ' % ctl['rms'] + ' | '.join('R25 %.2f tau %.2g EaR %.2f' % (p['R25'], p['tau25'], p['EaR']) for p in ctl['procs']))
    for v in C.values(): v['all'].pop('fun', None)
    json.dump(dict(A=A, C={'%s_%d_K%d' % k: v for k, v in C.items()}, control=ctl), open(os.path.join(HERE, 'cold_split.json'), 'w'), indent=1)
