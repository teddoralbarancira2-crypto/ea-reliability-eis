# -*- coding: utf-8 -*-
"""Stage 2: Arrhenius-coupled joint fit across all 10 temperatures, per (cell, SOC).
Z(T) = R0(T) + jwL(T) + sum_k ZARC(R_k(T), tau_k(T), a_k) + CPE tail(Q(T), n)
  R_k(T)   = R_k  exp( Ea_k /kB (1/T - 1/Tref))      (resistance label)
  tau_k(T) = tau_k exp( Et_k /kB (1/T - 1/Tref))     (time-constant label)
  Q(T)     = Q    exp(-Eq   /kB (1/T - 1/Tref))
R0, L free per temperature (nuisance). Compared with independent per-temperature fits of the same circuit.
f <= 2 kHz. Writes joint.json. py -3.11"""
import numpy as np, json, os, sys
from scipy.optimize import least_squares
from drt_all import load, CELLS, TEMPS, SOCS
HERE = os.path.dirname(os.path.abspath(__file__)); kB = 8.617333e-5; TREF = 298.15
TK = np.array(TEMPS) + 273.15; X = 1 / (kB * TK) - 1 / (kB * TREF)

def zarc(w, R, tau, a): return R / (1 + (1j * w * tau) ** a)
def model_T(p_proc, R0, L, w, K):
    z = R0 + 1j * w * L
    for k in range(K):
        R, tau, a = p_proc[3 * k:3 * k + 3]; z = z + zarc(w, R, tau, a)
    Q, n = p_proc[3 * K:3 * K + 2]; return z + 1 / (Q * (1j * w) ** n)

def unpack_joint(p, K, nT):
    R0 = p[:nT]; L = np.exp(p[nT:2 * nT]); q = p[2 * nT:]
    pr = []
    for k in range(K):
        lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]; pr.append((lR, Ea, lt, Et, a))
    lQ, Eq, n = q[5 * K:5 * K + 3]; return R0, L, pr, (lQ, Eq, n)

def resid_joint(p, data, K):
    R0, L, pr, (lQ, Eq, n) = unpack_joint(p, K, len(data)); out = []
    for i, (f, zr, zi) in enumerate(data):
        w = 2 * np.pi * f; pp = []
        for lR, Ea, lt, Et, a in pr: pp += [np.exp(lR + Ea * X[i]), np.exp(lt + Et * X[i]), a]
        pp += [np.exp(lQ - Eq * X[i]), n]
        z = model_T(pp, R0[i], L[i], w, K); out += [z.real - zr, -z.imag - zi]
    return np.concatenate(out)

def fit_joint(data, K, init_tau):
    nT = len(data)
    p0 = list(np.array([d[1][np.argmin(np.abs(d[0] - 1000))] for d in data]) * 0.98) + [np.log(5e-4)] * nT
    lo = [0] * nT + [np.log(1e-5)] * nT; hi = [50] * nT + [np.log(5e-3)] * nT
    for t in init_tau:
        p0 += [np.log(2.0), 0.4, np.log(t), 0.4, 0.8]; lo += [np.log(1e-3), -0.5, np.log(1e-6), -0.5, 0.4]; hi += [np.log(100), 1.5, np.log(50), 1.5, 1.0]
    p0 += [np.log(50.0), 0.3, 0.6]; lo += [np.log(0.1), -0.5, 0.3]; hi += [np.log(1e5), 1.5, 1.0]
    r = least_squares(resid_joint, p0, args=(data, K), bounds=(lo, hi), x_scale='jac', max_nfev=20000)
    J = r.jac; dof = len(r.fun) - len(p0); s2 = (r.fun ** 2).sum() / dof
    try: cov = np.linalg.pinv(J.T @ J) * s2; se = np.sqrt(np.diag(cov))
    except Exception: se = np.full(len(p0), np.nan)
    return r, se, np.sqrt(np.mean(r.fun ** 2))

def summarize(r, se, K, nT):
    q = r.x[2 * nT:]; sq = se[2 * nT:]; out = []
    for k in range(K):
        lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]
        out.append(dict(R25=float(np.exp(lR)), Ea=float(Ea), Ea_se=float(sq[5 * k + 1]), tau25=float(np.exp(lt)), Et=float(Et), Et_se=float(sq[5 * k + 3]), a=float(a)))
    out.sort(key=lambda d: d['tau25']); return out

if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    INIT = {2: [1e-3, 2e-2], 3: [5e-4, 1e-2, 0.2]}[K]
    res = {}
    for c, cell in enumerate(CELLS):
        for soc in SOCS:
            # unit convention: mOhm
            data = [(f, zr * 1e3, zi * 1e3) for f, zr, zi in (load(cell, t, soc) for t in TEMPS)]
            r, se, rms = fit_joint(data, K, INIT)
            S = summarize(r, se, K, len(data)); res['%s_%d' % (cell, soc)] = dict(rms=rms, proc=S)
            print('%s SOC%3d rms %.4f mOhm | ' % (cell, soc, rms) + ' | '.join(
                'R25 %.2f tau25 %.2g Ea_R %.3f±%.3f Ea_tau %.3f±%.3f a %.2f' % (d['R25'], d['tau25'], d['Ea'], d['Ea_se'], d['Et'], d['Et_se'], d['a']) for d in S), flush=True)
    json.dump(res, open(os.path.join(HERE, 'joint_K%d.json' % K), 'w'), indent=1)
