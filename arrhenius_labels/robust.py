# -*- coding: utf-8 -*-
"""Stage 5 robustness, SOC 50/70, both cells, joint 2-process fit started from joint_ms_K2.json:
 (1) temperature jackknife (drop one T, refit) -> spread of dEa_R, dEa_tau
 (2) per-temperature weighting (residuals / polarization span of each spectrum)
 (3) hold-out: fit {0,15,30,45} C, predict the other six spectra and their apparent Ea(f)
 (4) identifiability: SE of R1, R2 from independent per-T fits vs the joint fit
Writes robust.json. py -3.11"""
import numpy as np, json
from scipy.optimize import least_squares
from drt_all import load, TEMPS
import joint_fit as J
R = json.load(open('joint_ms_K2.json')); nT = 10; kB = J.kB; TREF = J.TREF

def resid_sub(p, data, K, xs, wts):
    R0 = p[:len(data)]; L = np.exp(p[len(data):2 * len(data)]); q = p[2 * len(data):]; out = []
    for i, (f, zr, zi) in enumerate(data):
        w = 2 * np.pi * f; pp = []
        for k in range(K):
            lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]; pp += [np.exp(lR + Ea * xs[i]), np.exp(lt + Et * xs[i]), a]
        lQ, Eq, n = q[5 * K:5 * K + 3]; pp += [np.exp(lQ - Eq * xs[i]), n]
        z = J.model_T(pp, R0[i], L[i], w, K); out += [(z.real - zr) / wts[i], (-z.imag - zi) / wts[i]]
    return np.concatenate(out)

def fit_sub(x_full, data_all, keep, wts=None):
    data = [data_all[i] for i in keep]; xs = J.X[keep]; wts = np.ones(len(keep)) if wts is None else wts
    p0 = np.concatenate([x_full[keep], x_full[nT + np.array(keep)], x_full[2 * nT:]])
    r = least_squares(resid_sub, p0, args=(data, 2, xs, wts), x_scale='jac', max_nfev=4000)
    q = r.x[2 * len(keep):]; big = sorted([(np.exp(q[0]), 0), (np.exp(q[5]), 5)], reverse=True)
    i1, i2 = (0, 5) if np.exp(q[2]) < np.exp(q[7]) else (5, 0)          # fast, slow by tau25
    return r, dict(EaR_f=q[i1 + 1], EaR_s=q[i2 + 1], Eat_f=q[i1 + 3], Eat_s=q[i2 + 3], dEaR=q[i2 + 1] - q[i1 + 1], dEat=q[i2 + 3] - q[i1 + 3])

def zarc_fit_single(f, zr, zi, x0):   # independent per-T fit, same circuit, returns R1, R2 and their SE
    def res(p):
        w = 2 * np.pi * f; z = p[0] + 1j * w * np.exp(p[1]) + J.zarc(w, np.exp(p[2]), np.exp(p[3]), p[4]) + J.zarc(w, np.exp(p[5]), np.exp(p[6]), p[7]) + 1 / (np.exp(p[8]) * (1j * w) ** p[9])
        return np.concatenate([z.real - zr, -z.imag - zi])
    r = least_squares(res, x0, x_scale='jac', max_nfev=4000)
    s2 = (r.fun ** 2).sum() / (len(r.fun) - len(x0)); cov = np.linalg.pinv(r.jac.T @ r.jac) * s2
    R1, R2 = np.exp(r.x[2]), np.exp(r.x[5]); return R1, R2, R1 * np.sqrt(cov[2, 2]), R2 * np.sqrt(cov[5, 5])

out = {}
for cell in ('no1_ch3', 'no2_ch4'):
    for soc in (50, 70):
        key = '%s_%d' % (cell, soc); x = np.array(R[key]['x'])
        data = [(f, zr * 1e3, zi * 1e3) for f, zr, zi in (load(cell, t, soc) for t in TEMPS)]
        full = fit_sub(x, data, list(range(nT)))[1]
        jk = [fit_sub(x, data, [i for i in range(nT) if i != d])[1] for d in range(nT)]
        span = np.array([d[1].max() - d[1].min() for d in data]); ww = fit_sub(x, data, list(range(nT)), wts=span / span.mean())[1]
        keep = [0, 3, 6, 9]; rH, hh = fit_sub(x, data, keep)
        # predict held-out spectra with the 4-T fit: R0/L of held-out taken from the full fit (nuisance), labels from the 4-T fit
        q = rH.x[2 * len(keep):]; xp = np.concatenate([x[:2 * nT], q]); res_all = J.resid_joint(xp, data, 2)
        nf = len(data[0][0]); held = [i for i in range(nT) if i not in keep]
        rms_held = float(np.sqrt(np.mean(np.concatenate([res_all[2 * nf * i:2 * nf * (i + 1)] for i in held]) ** 2)))
        span_held = float(np.mean(span[held]))
        # identifiability: per-T SE of R1, R2 vs joint SE (propagated at each T)
        q0 = x[2 * nT:]; se_single = []
        for i, (f, zr, zi) in enumerate(data):
            x0 = [x[i], x[nT + i], q0[0] + q0[1] * J.X[i], q0[2] + q0[3] * J.X[i], q0[4], q0[5] + q0[6] * J.X[i], q0[7] + q0[8] * J.X[i], q0[9], q0[10] - q0[11] * J.X[i], q0[12]]
            R1, R2, s1, s2 = zarc_fit_single(f, zr, zi, x0); se_single.append((s1 / R1, s2 / R2))
        rj = least_squares(J.resid_joint, x, args=(data, 2), x_scale='jac', max_nfev=10)
        s2 = (rj.fun ** 2).sum() / (len(rj.fun) - len(x)); cov = np.linalg.pinv(rj.jac.T @ rj.jac) * s2
        se_joint = []
        for i in range(nT):
            for b in (2 * nT, 2 * nT + 5):
                g = np.zeros(len(x)); g[b] = 1; g[b + 1] = J.X[i]; se_joint.append(np.sqrt(g @ cov @ g))
        se_joint = np.array(se_joint).reshape(nT, 2); se_single = np.array(se_single)
        o = dict(full=full, jk_dEaR=[float(d['dEaR']) for d in jk], jk_dEat=[float(d['dEat']) for d in jk],
                 jk_EaR_f=[float(d['EaR_f']) for d in jk], jk_EaR_s=[float(d['EaR_s']) for d in jk],
                 weighted=ww, hold4=hh, rms_held=rms_held, span_held=span_held,
                 relSE_single_med=list(map(float, np.median(se_single, axis=0))), relSE_joint_med=list(map(float, np.median(se_joint, axis=0))))
        o = json.loads(json.dumps(o, default=float)); out[key] = o
        jr = np.array(o['jk_dEaR']); jt = np.array(o['jk_dEat']); n = nT
        print('%s full dEaR %.3f dEat %.3f | jackknife dEaR %.3f±%.3f [%.3f..%.3f] dEat %.3f±%.3f [%.3f..%.3f] | weighted dEaR %.3f dEat %.3f | 4T fit dEaR %.3f dEat %.3f, held-out rms %.3f mOhm (arc span %.2f) | relSE R1,R2 single %s joint %s' % (
            key, full['dEaR'], full['dEat'], jr.mean(), np.sqrt((n - 1) / n * ((jr - jr.mean()) ** 2).sum()), jr.min(), jr.max(),
            jt.mean(), np.sqrt((n - 1) / n * ((jt - jt.mean()) ** 2).sum()), jt.min(), jt.max(), ww['dEaR'], ww['dEat'], hh['dEaR'], hh['dEat'], rms_held, span_held,
            np.round(o['relSE_single_med'], 3), np.round(o['relSE_joint_med'], 3)), flush=True)
json.dump(out, open('robust.json', 'w'), indent=1)
