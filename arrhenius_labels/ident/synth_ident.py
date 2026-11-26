# -*- coding: utf-8 -*-
"""Use-3 artifact test: is the slow-process Ea failure on dataset B explained by process proximity (tau ratio) and size,
or is it something else? Synthetic two-ZARC + CPE spectra on the dataset B design (10 T 0-45 C, dataset B f grid <= 2 kHz),
truth = dataset B joint K=2 fit, cell no1_ch3 SOC 50 (fast R25 3.15 mOhm, tau 0.41 ms, a 0.65, Ea_R 0.44, Ea_tau 0.50;
slow R25 0.96, tau 2.9 ms, a 0.92, Ea_R 0.46, Ea_tau 0.59). Varied: tau ratio r = tau_slow/tau_fast, slow/fast size
ratio m, noise sigma. Two truths: (W) well specified; (M) mildly misspecified, fast-arc alpha(T) = 0.65 - 0.015 X
(0.60 at 0 C, 0.68 at 45 C), fitted with constant alpha. Per case: Cramer-Rao SD of slow Ea_R from the Jacobian at
truth, Monte-Carlo fits (all T, and dropping 0 C, the dataset B swing test). Writes synth_ident.json. py -3.11"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drt_all import load, TEMPS
kB = 8.617333e-5; TREF = 298.15
TT = np.array(TEMPS, float); XS = 1 / (kB * (TT + 273.15)) - 1 / (kB * TREF)
F = load('no1_ch3', 25, 50)[0]
# truth nuisance terms = the real joint K=2 fit of no1_ch3 SOC 50 (rms 0.058 mOhm; per-T 0.03-0.10)
R0T = np.array([13.721, 13.781, 13.598, 13.48, 13.351, 13.443, 13.401, 13.442, 13.476, 13.545])
FAST = (1.1462, 0.4414, -7.81, 0.4983, 0.6542); SLOW = (0.4646, 0.5926, 0.9211)   # slow: (Ea_R, Ea_tau, alpha)
TAIL = (-0.2194, 0.0308, 0.604); L0 = 5.2e-4      # tail ln Q, Eq, n (mOhm units); L in mOhm*s
RATIOS = [3, 5, 7, 10, 20, 40, 100]; SIZES = [0.3, 1.0, 3.0]; SIGMAS = [0.015, 0.06]; NSEED = 8

def model(q, xs, w, a_slope=0.0):
    """q = [R0 per T..., lL, 5 fast, 5 slow, 3 tail]; returns list of complex Z (mOhm)."""
    n = len(xs); R0 = q[:n]; L = np.exp(q[n]); p = q[n + 1:]; out = []
    for i, x in enumerate(xs):
        z = R0[i] + 1j * w * L
        for k in range(2):
            lR, Ea, lt, Et, a = p[5 * k:5 * k + 5]
            if k == 0: a = a + a_slope * x
            z = z + np.exp(lR + Ea * x) / (1 + (1j * w * np.exp(lt + Et * x)) ** a)
        lQ, Eq, nn = p[10:13]; z = z + 1 / (np.exp(lQ - Eq * x) * (1j * w) ** nn); out.append(z)
    return out

def resid(q, xs, w, data):
    return np.concatenate([np.concatenate([z.real - d.real, z.imag - d.imag]) for z, d in zip(model(q, xs, w), data)])

def truth(r, m):
    lt_s = FAST[2] + np.log(r); lR_s = FAST[0] + np.log(m)
    return np.array(list(R0T) + [np.log(L0)] + list(FAST) + [lR_s, SLOW[0], lt_s, SLOW[1], SLOW[2]] + list(TAIL))

def bounds(n):
    lo = [0] * n + [np.log(1e-9)] + [np.log(1e-3), -0.5, np.log(1e-6), -0.5, 0.4] * 2 + [np.log(0.1), -0.5, 0.3]
    hi = [50] * n + [np.log(1e-2)] + [np.log(100), 1.5, np.log(10), 1.5, 1.0] * 2 + [np.log(1e5), 1.5, 1.0]
    return np.array(lo, float), np.array(hi, float)

def fit(data, xs, w, q0, rng):
    n = len(xs); lo, hi = bounds(n); best = None
    starts = [q0]
    for _ in range(3):                           # perturbed starts (labels +-0.2 eV, sizes x/2..x2, taus x/3..x3)
        s = q0.copy(); p = s[n + 1:]
        for k in range(2):
            p[5 * k] += rng.uniform(-0.7, 0.7); p[5 * k + 1] += rng.uniform(-0.2, 0.2); p[5 * k + 2] += rng.uniform(-1.1, 1.1); p[5 * k + 3] += rng.uniform(-0.2, 0.2)
        starts.append(s)
    for s in starts:
        s = np.clip(s, lo + 1e-6, hi - 1e-6)
        try: rr = least_squares(resid, s, args=(xs, w, data), bounds=(lo, hi), x_scale='jac', max_nfev=6000)
        except Exception: continue
        if best is None or rr.cost < best.cost: best = rr
    p = best.x[n + 1:]
    k = int(np.argmax([p[2], p[7]]))              # slow = longer tau
    return float(p[5 * k + 1]), float(p[5 * k + 3]), float(np.exp(p[5 * k]))

def crb(q, xs, w, sigma, eps=1e-6):
    base = np.concatenate([np.concatenate([z.real, z.imag]) for z in model(q, xs, w)]); J = []
    for i in range(len(q)):
        dq = q.copy(); dq[i] += eps
        J.append((np.concatenate([np.concatenate([z.real, z.imag]) for z in model(dq, xs, w)]) - base) / eps)
    J = np.array(J).T; cov = np.linalg.pinv(J.T @ J) * sigma ** 2; n = len(xs)
    i_eaR = n + 1 + 6; i_eat = n + 1 + 8
    return float(np.sqrt(cov[i_eaR, i_eaR])), float(np.sqrt(cov[i_eat, i_eat]))

def job(args):
    r, m, sig, mode = args; w = 2 * np.pi * F; q = truth(r, m); a_sl = {'W': 0.0, 'M': -0.015, 'M2': -0.03}[mode]
    rng = np.random.default_rng(int(r * 1000 + m * 10 + sig * 1e4) + {'W': 0, 'M': 7, 'M2': 13}[mode])
    clean = model(q, XS, w, a_sl); out = dict(r=r, m=m, sigma=sig, mode=mode)
    out['crb_EaR'], out['crb_Eat'] = crb(q, XS, w, sig)
    full, drop0 = [], []
    for s in range(NSEED):
        data = [z + rng.normal(0, sig, len(w)) + 1j * rng.normal(0, sig, len(w)) for z in clean]
        full.append(fit(data, XS, w, q, rng))
        keep = TT > 0
        drop0.append(fit([d for d, kk in zip(data, keep) if kk], XS[keep], w, np.r_[q[:len(XS)][keep], q[len(XS):]], rng))
    out['full'] = full; out['drop0'] = drop0
    return out

if __name__ == '__main__':
    jobs = [(r, m, s, mode) for mode in ('W', 'M', 'M2') for r in RATIOS for m in SIZES for s in SIGMAS]
    with Pool(18) as pool: res = pool.map(job, jobs)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth_ident.json'), 'w'), indent=1)
    for o in res:
        f = np.array(o['full']); d = np.array(o['drop0'])
        print('%-2s r %5.0f m %.1f sig %.3f | CRB sd EaR %.3f | fit EaR %.3f +- %.3f (truth 0.46) | drop0 %.3f | swing %.3f' % (
            o['mode'], o['r'], o['m'], o['sigma'], o['crb_EaR'], f[:, 0].mean(), f[:, 0].std(), d[:, 0].mean(), abs(d[:, 0].mean() - f[:, 0].mean())), flush=True)
