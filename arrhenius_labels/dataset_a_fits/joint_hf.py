# -*- coding: utf-8 -*-
"""Joint Arrhenius-coupled fit across temperature at every checkpoint, with the fast arc kept inside the window.
Variants (SOC step S, cold multistart, every checkpoint):
 A  f <= 10 kHz, lead = jwL1(T) + (jwL2*RL)/(RL + jwL2) with L2, RL shared over T; R0 free per T
 B  as A, but R0(T) = R0_25 * (1 + b (T - 25))                     (2 shared parameters)
 C  f <= 2 kHz, plain jwL(T), R0(T) linear as in B
 D  as B, plus a temperature-dependent slow-arc shape: alpha_slow(T) = a0 + a1 * X   (X = 1/kT - 1/kTref)
 E  as B, slow arc Havriliak-Negami R/(1+(jw tau)^a)^b (asymmetric, b free in 0.3-1)
Processes: 3 ZARCs in tau25 windows + CPE tail. fit_data() is also used by the synthetic leakage test (synth_leak.py).
Writes hf_{V}_s{S}.json. py -3.11 joint_hf.py V S"""
import numpy as np, json, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
from common import Z, F, TS, CYC, ok, JS
kB = 8.617333e-5; TREF = 298.15; K = 3
TAUB = [(1e-6, 1.5e-3), (1.5e-3, 1.5e-2), (1.5e-2, 2.0)]
SEED = [(1.0, 0.40, 3e-4, 0.40, 0.7), (2.0, 0.40, 3e-3, 0.50, 0.8), (2.0, 0.55, 5e-2, 0.55, 0.9)]

class Setup:
    def __init__(self, V):
        self.V = V; self.FMAX = 2000.0 if V == 'C' else 10001.0
        self.HF = V in 'ABDE'; self.R0LIN = V in 'BCDE'; self.SLOPEA = V == 'D'; self.HN = V == 'E'; self.MF = F <= self.FMAX * 1.001

def unpack(p, n, st, TT):
    i = 0
    if st.R0LIN: R0 = p[0] * (1 + p[1] * (TT - 25)); i = 2
    else: R0 = p[:n]; i = n
    L1 = np.exp(p[i:i + n]); i += n
    if st.HF: L2, RL = np.exp(p[i]), np.exp(p[i + 1]); i += 2
    else: L2 = RL = None
    return R0, L1, L2, RL, p[i:]

def model(p, f, xs, TT, st):
    n = len(xs); R0, L1, L2, RL, q = unpack(p, n, st, TT); out = []
    for i in range(n):
        w = 2 * np.pi * f; z = R0[i] + 1j * w * L1[i]
        if st.HF: z = z + (1j * w * L2 * RL) / (RL + 1j * w * L2)
        for k in range(K):
            lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]
            if k == 2 and st.SLOPEA: a = np.clip(a + q[5 * K + 3] * xs[i], 0.3, 1.0)
            b = q[5 * K + 3] if (k == 2 and st.HN) else 1.0
            z = z + np.exp(lR + Ea * xs[i]) / (1 + (1j * w * np.exp(lt + Et * xs[i])) ** a) ** b
        lQ, Eq, nn = q[5 * K:5 * K + 3]; z = z + 1 / (np.exp(lQ - Eq * xs[i]) * (1j * w) ** nn)
        out.append(z)
    return out

def resid(p, data, xs, TT, st):
    zs = model(p, data[0][0], xs, TT, st)
    return np.concatenate([np.concatenate([z.real - zr, -z.imag - zi]) for z, (f, zr, zi) in zip(zs, data)])

def fit_data(data, TT, st, seed=0, n_rand=5):
    n = len(data); xs = 1 / (kB * (TT + 273.15)) - 1 / (kB * TREF)
    r0g = np.array([d[1][np.argmin(np.abs(d[0] - 1500))] for d in data]) * 0.97
    if st.R0LIN: head, lo, hi = [r0g.mean(), 0.0], [0, -0.02], [100, 0.02]
    else: head, lo, hi = list(r0g), [0] * n, [100] * n
    head += [np.log(3e-6 if st.HF else 1e-5)] * n; lo += [np.log(1e-9)] * n; hi += [np.log(1e-3)] * n
    if st.HF: head += [np.log(3e-6), np.log(10.0)]; lo += [np.log(1e-9), np.log(0.01)]; hi += [np.log(1e-3), np.log(1e3)]
    plo, phi = [], []
    for tl, th in TAUB: plo += [np.log(1e-3), -0.5, np.log(tl), -0.5, 0.4]; phi += [np.log(500), 2.0, np.log(th), 2.0, 1.0]
    plo += [np.log(0.01), -0.5, 0.3]; phi += [np.log(1e5), 2.0, 1.0]
    tail0 = [np.log(20.0), 0.4, 0.6]
    if st.SLOPEA: plo += [-0.1]; phi += [0.1]
    if st.HN: plo += [0.3]; phi += [1.0]
    lo = np.array(lo + plo); hi = np.array(hi + phi); rng = np.random.default_rng(seed)
    ext = [0.0] if st.SLOPEA else ([0.8] if st.HN else [])
    starts = [head + sum([[np.log(R), Ea, np.log(t), Et, a_] for R, Ea, t, Et, a_ in SEED], []) + tail0 + ext]
    for _ in range(n_rand):
        p = list(head)
        for tl, th in TAUB: p += [np.log(rng.uniform(0.3, 8)), rng.uniform(0.2, 0.8), rng.uniform(np.log(tl), np.log(th)), rng.uniform(0.2, 0.8), rng.uniform(0.5, 0.95)]
        p += [np.log(rng.uniform(5, 100)), rng.uniform(0.1, 0.7), rng.uniform(0.4, 0.8)] + ([rng.uniform(-0.05, 0.05)] if st.SLOPEA else ([rng.uniform(0.5, 1.0)] if st.HN else [])); starts.append(p)
    best = None
    for p0 in starts:
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        try: r = least_squares(resid, p0, args=(data, xs, TT, st), bounds=(lo, hi), x_scale='jac', max_nfev=8000)
        except Exception: continue
        if best is None or r.cost < best.cost: best = r
    R0, L1, L2, RL, q = unpack(best.x, n, st, TT); nh = len(best.x) - len(q); flags = []
    for i, (v, a_, b_) in enumerate(zip(best.x, lo, hi)):
        if (v - a_) <= 0.01 * (b_ - a_) or (b_ - v) <= 0.01 * (b_ - a_):
            flags.append('h%d' % i if i < nh else ('p%d.%d' % ((i - nh) // 5, (i - nh) % 5) if i - nh < 5 * K else 'x%d' % (i - nh - 5 * K)))
    return dict(rms=float(np.sqrt(np.mean(best.fun ** 2))), q=list(map(float, q)), R0=list(map(float, R0)), x=list(map(float, best.x)),
                L2=None if L2 is None else float(L2), RL=None if RL is None else float(RL), flags=flags)

def fast_band_ea(q, TT=None):
    T = np.array([0, 5, 15, 25, 35, 45, 55]) + 273.15 if TT is None else TT + 273.15
    x = 1 / (kB * T); X = x - 1 / (kB * TREF); y = sum(np.exp(q[5 * k] + q[5 * k + 1] * X) for k in (0, 1))
    return np.polyfit(x, np.log(y), 1)[0], y[np.argmin(abs(T - 298.15))]

def job(a):
    c, j, V, S = a; st = Setup(V); ks = [k for k in range(7) if ok(c, j, k)]; TT = TS[ks].astype(float)
    data = [(F[st.MF], Z[c, j, k, S][st.MF].real * 1e3, -Z[c, j, k, S][st.MF].imag * 1e3) for k in ks]
    out = fit_data(data, TT, st, seed=1000 * c + j); out.update(cyc=int(CYC[j]), temps=[int(t) for t in TT]); return (c, j), out

if __name__ == '__main__':
    V = sys.argv[1]; S = int(sys.argv[2])
    with Pool(8) as pool: res = pool.map(job, [(c, j, V, S) for c in (0, 1) for j in JS])
    js = {'cell%d_%d' % (c + 1, CYC[j]): v for (c, j), v in res}
    json.dump(js, open('hf_%s_s%d.json' % (V, S), 'w'), indent=1)
    X = 1 / (kB * (TS + 273.15)) - 1 / (kB * TREF)
    for key in sorted(js, key=lambda k: (k[:5], int(k.split('_')[1]))):
        v = js[key]; q = v['q']; e, fb25 = fast_band_ea(q)
        fp = 1 / (2 * np.pi * np.exp(q[2] + q[3] * X)); ex = ' a1 %.3f (alpha_slow 0/55C %.2f/%.2f)' % (q[18], q[14] + q[18] * X[0], q[14] + q[18] * X[6]) if V == 'D' else (' beta_slow %.2f' % q[18] if V == 'E' else '')
        print('%-10s rms %.3f | slow R25 %.2f EaR %.3f Eat %.3f | fast R25 %.2f EaR %.2f | mid R25 %.2f EaR %.2f Eat %.2f | fast-band Ea_eff %.3f | fast peak 55C %6.0f Hz%s | flags %s' % (
            key, v['rms'], np.exp(q[10]), q[11], q[13], np.exp(q[0]), q[1], np.exp(q[5]), q[6], q[8], e, fp[6], ex, ','.join(v['flags'])), flush=True)
