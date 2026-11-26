# -*- coding: utf-8 -*-
"""Consequence test: does an activation energy that fails the window test cause practical error?
Use case: calibrate the Arrhenius-coupled impedance model on the temperatures available, predict the MEASURED impedance
at temperatures that were not used (what a BMS temperature-compensation table or Arrhenius ECM does).
Metric: relative error of Re Z at 1 Hz and 0.1 Hz at each held-out T, and its equivalent temperature error (error in
ln Re Z divided by the measured local slope d ln Re Z / dT). Baseline: a lumped Arrhenius fit of Re Z(f) itself.
Cases: dataset B (K=2, flagged unreliable) both cells SOC 50/70; dataset A variant B at 3.73 V, cycles 120-480
(window test fails at 120-200, passes from 240). Splits: warm-calibrated (predict cold) and cold-calibrated (predict hot).
Writes consequence.json. py -3.11"""
import numpy as np, json, os, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'dataset_a_fits'))
kB = 8.617333e-5; TREF = 298.15
FQ = (10.0, 1.0, 0.1)

def xs_of(T): return 1 / (kB * (np.asarray(T, float) + 273.15)) - 1 / (kB * TREF)

def at(f, z, fq): return z.real[np.argmin(np.abs(f - fq))]

def linear(Tc, y, Tp):
    """Schinagl et al. 2025 linear method: Re Z = a + k T per frequency."""
    k, a = np.polyfit(np.asarray(Tc, float), y, 1); return a + k * np.asarray(Tp, float)

def lut(Tc, y, Tp):
    """Lookup table: piecewise-linear interpolation of ln Re Z in 1/T between the nearest calibration temperatures
    (end segments extended for extrapolation). The strongest per-frequency competitor."""
    x = xs_of(Tc); o = np.argsort(x); x, ly = x[o], np.log(np.asarray(y)[o]); xp = xs_of(Tp); out = []
    for v in np.atleast_1d(xp):
        i = np.clip(np.searchsorted(x, v) - 1, 0, len(x) - 2)
        out.append(ly[i] + (ly[i + 1] - ly[i]) * (v - x[i]) / (x[i + 1] - x[i]))
    return np.exp(out)

def lumped(Tc, y, Tp):
    """Arrhenius of Re Z itself: ln y = a + b x, fitted on calibration T, predicted at Tp."""
    b, a = np.polyfit(xs_of(Tc), np.log(y), 1); return np.exp(a + b * xs_of(Tp))

def score(f, zmeas, zpred, Tp, slope):
    out = {}
    for fq in FQ:
        m = np.array([at(f, z, fq) for z in zmeas]); p = np.array([at(f, z, fq) for z in zpred])
        rel = (p - m) / m; out['rel_%g' % fq] = rel.tolist(); out['dT_%g' % fq] = (np.log(p / m) / slope[fq]).tolist()
    return out

# ---------------- dataset B ----------------
def dsb_job(a):
    cell, soc, split = a
    from cold_split import getdata, fit as fit_dsb, model as model_dsb
    from drt_all import TEMPS
    w, data = getdata(cell, soc); f = w / 2 / np.pi; TT = np.array(TEMPS, float)
    cal = {'warm': TT >= 10, 'cold': TT <= 30, 'interp': np.isin(TT, [0, 10, 20, 30, 45])}[split]; prd = ~cal
    r = None
    import cold_split as CS
    n = cal.sum(); xs = CS.XS[cal]
    # refit to get the parameter vector (fit() returns summaries only) -> local copy of the optimizer call
    from scipy.optimize import least_squares
    best = None; rng = np.random.default_rng(7)
    sub = [d for d, k in zip(data, cal) if k]
    r0 = [d.real[np.argmin(np.abs(f - 1000))] * 0.98 for d in sub]
    lo = [0] * n + [np.log(1e-6)] * n + [np.log(1e-3), -0.5, np.log(1e-5), -0.5, 0.4] * 2 + [np.log(0.01), -0.5, 0.3]
    hi = [50] * n + [np.log(5e-3)] * n + [np.log(100), 1.5, np.log(0.1), 1.5, 1.0] * 2 + [np.log(1e5), 1.5, 1.0]
    lo, hi = np.array(lo), np.array(hi)
    for s in range(8):
        p = list(r0) + [np.log(5e-4)] * n
        for t in (4e-4, 3e-3):
            p += [np.log(2.0 if s == 0 else rng.uniform(0.3, 3)), 0.45 if s == 0 else rng.uniform(0.2, 0.8), np.log(t * (1 if s == 0 else np.exp(rng.uniform(-1, 1)))), 0.5 if s == 0 else rng.uniform(0.2, 0.8), 0.8 if s == 0 else rng.uniform(0.5, 0.95)]
        p += [np.log(0.8), 0.05, 0.6]; p = np.clip(p, lo + 1e-6, hi - 1e-6)
        rr = least_squares(CS.resid, p, args=(xs, w, sub, 2), bounds=(lo, hi), x_scale='jac', max_nfev=8000)
        if best is None or rr.cost < best.cost: best = rr
    q = best.x; R0c = q[:n]; Lc = q[n:2 * n]; proc = q[2 * n:]
    # held-out R0: linear in T from calibration R0; L: mean
    bR, aR = np.polyfit(TT[cal], R0c, 1); xp = CS.XS[prd]
    qp = np.r_[aR + bR * TT[prd], np.full(prd.sum(), Lc.mean()), proc]
    zpred = CS.model(qp, xp, w, 2); zmeas = [d for d, k in zip(data, prd) if k]
    slope = {fq: np.polyfit(TT, np.log([at(f, z, fq) for z in data]), 1)[0] for fq in FQ}   # local d lnReZ/dT (1/K), all T
    out = score(f, zmeas, zpred, TT[prd], slope)
    for fq in FQ:
        y = np.array([at(f, z, fq) for z in sub]); m = np.array([at(f, z, fq) for z in zmeas])
        p = lumped(TT[cal], y, TT[prd]); out['base_rel_%g' % fq] = ((p - m) / m).tolist(); pl = linear(TT[cal], y, TT[prd]); out['lin_rel_%g' % fq] = ((pl - m) / m).tolist(); pt = lut(TT[cal], y, TT[prd]); out['lut_rel_%g' % fq] = ((pt - m) / m).tolist()
    k = int(np.argmax([proc[2], proc[7]])); out.update(case='B %s SOC%d' % (cell, soc), split=split, Tpred=TT[prd].tolist(), slowEa=float(proc[5 * k + 1]), flag='fail')
    return out

# ---------------- dataset A ----------------
def auto_job(a):
    c, n, split = a
    from common import Z, F, TS, CYC, ok
    import joint_hf as JH
    st = JH.Setup('B'); j = CYC.index(n); ks = [k for k in range(7) if ok(c, j, k)]; TT = TS[ks].astype(float)
    f = F[st.MF]; data = [Z[c, j, k, 10][st.MF] * 1e3 for k in ks]
    cal = {'warm': TT >= 15, 'cold': TT <= 35, 'interp': np.isin(TT, [0, 15, 35, 55])}[split]; prd = ~cal
    sub = [(f, z.real, -z.imag) for z, kk in zip(data, cal) if kk]
    r = JH.fit_data(sub, TT[cal], st, seed=31 * c + j, n_rand=4); x = np.array(r['x']); m = cal.sum()
    R0a, R0b = x[0], x[1]; L1 = x[2:2 + m]; rest = x[2 + m:]
    qp = np.r_[R0a, R0b, np.full(prd.sum(), L1.mean()), rest]
    zpred = JH.model(qp, f, xs_of(TT[prd]), TT[prd], st); zmeas = [z for z, kk in zip(data, prd) if kk]
    slope = {fq: np.polyfit(TT, np.log([at(f, z, fq) for z in data]), 1)[0] for fq in FQ}
    out = score(f, zmeas, zpred, TT[prd], slope)
    for fq in FQ:
        y = np.array([at(f, z, fq) for z, kk in zip(data, cal) if kk]); mm = np.array([at(f, z, fq) for z in zmeas])
        p = lumped(TT[cal], y, TT[prd]); out['base_rel_%g' % fq] = ((p - mm) / mm).tolist(); pl = linear(TT[cal], y, TT[prd]); out['lin_rel_%g' % fq] = ((pl - mm) / mm).tolist(); pt = lut(TT[cal], y, TT[prd]); out['lut_rel_%g' % fq] = ((pt - mm) / mm).tolist()
    WS = json.load(open(os.path.join(ROOT, 'dataset_a_fits', 'window_scan_s10.json')))
    sp = [WS['%d_%d_%d-%d' % (c + 1, n, a_, b_)]['EaR'] for a_, b_ in [(0, 25), (5, 35), (15, 45), (25, 55), (0, 55)] if '%d_%d_%d-%d' % (c + 1, n, a_, b_) in WS]
    span = max(sp) - min(sp) if sp else np.nan
    out.update(case='A cell%d cyc%d' % (c + 1, n), split=split, Tpred=TT[prd].tolist(), slowEa=float(rest[5 * 2 + 1]), span=float(span), flag='pass' if span <= 0.072 else 'fail')
    return out

if __name__ == '__main__':
    os.chdir(os.path.join(ROOT, 'dataset_a_fits'))
    jobs_l = [(c, s, sp) for c in ('no1_ch3', 'no2_ch4') for s in (50, 70) for sp in ('warm', 'cold', 'interp')]
    jobs_a = [(c, n, sp) for c in (0, 1) for n in (120, 160, 200, 240, 280, 320, 360, 480) for sp in ('warm', 'cold', 'interp')]
    with Pool(16) as pool:
        RL = pool.map(dsb_job, jobs_l); RA = pool.map(auto_job, jobs_a)
    json.dump(RL + RA, open(os.path.join(HERE, 'consequence2.json'), 'w'), indent=1)
    for o in RL + RA:
        f = lambda k: ' '.join('%+.1f' % (100 * v) for v in o[k])
        print('%-22s %-6s %s | Tpred %s | 10Hz ours %s lin %s arr %s | 1Hz ours %s lin %s arr %s | 0.1Hz ours %s lin %s arr %s' % (o['case'], o['split'], o['flag'], o['Tpred'],
              f('rel_10'), f('lin_rel_10'), f('base_rel_10'), f('rel_1'), f('lin_rel_1'), f('base_rel_1'), f('rel_0.1'), f('lin_rel_0.1'), f('base_rel_0.1')), flush=True)
