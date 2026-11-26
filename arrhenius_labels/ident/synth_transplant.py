# -*- coding: utf-8 -*-
"""Follow-up to synth_ident: transplant the REAL dataset B misfit into synthetic spectra. The joint K=2 fit residual of
no1_ch3 SOC 50 (per T, real and imaginary) is added to the clean synthetic spectra, either at every temperature
('all') or at 0 and 5 C only ('cold'), small noise 0.015 mOhm on top. If the slow-Ea swing (full vs drop 0 C vs drop 5 C)
reappears at dataset B geometry (r 7, m 0.3) and fades for large/separated arcs, the dataset B failure is a cold-end misfit
whose leverage scales with arc size, not a tau-ratio limit. Writes synth_transplant.json. py -3.11"""
import numpy as np, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joint_fit as J
from drt_all import load, TEMPS
import synth_ident as S

def real_resid():
    data = [(f, zr * 1e3, zi * 1e3) for f, zr, zi in (load('no1_ch3', t, 50) for t in TEMPS)]
    r, _, _ = J.fit_joint(data, 2, [1e-3, 2e-2]); nf = len(data[0][0]); E = []
    for i in range(len(TEMPS)):
        er = r.fun[2 * i * nf:2 * i * nf + nf]; ei = r.fun[2 * i * nf + nf:2 * (i + 1) * nf]
        E.append(-er + 1j * ei)                  # data - model, as complex impedance error (mOhm)
    return E
E = real_resid()

def job(args):
    r, m, where = args; w = 2 * np.pi * S.F; q = S.truth(r, m); rng = np.random.default_rng(int(r * 100 + m * 10) + (1 if where == 'cold' else 0))
    clean = S.model(q, S.XS, w); out = dict(r=r, m=m, where=where, full=[], drop0=[], drop5=[])
    for s in range(4):
        data = []
        for i, z in enumerate(clean):
            e = E[i] if (where == 'all' or S.TT[i] <= 5) else 0
            data.append(z + e + rng.normal(0, 0.015, len(w)) + 1j * rng.normal(0, 0.015, len(w)))
        out['full'].append(S.fit(data, S.XS, w, q, rng))
        for key, drop in (('drop0', 0), ('drop5', 5)):
            keep = S.TT != drop
            out[key].append(S.fit([d for d, k in zip(data, keep) if k], S.XS[keep], w, np.r_[q[:len(S.XS)][keep], q[len(S.XS):]], rng))
    return out

if __name__ == '__main__':
    print('real residual rms per T:', [round(float(np.sqrt(np.mean(np.abs(e) ** 2) / 2)), 3) for e in E])
    jobs = [(r, m, wh) for wh in ('all', 'cold') for r in (3, 7, 20, 40, 100) for m in (0.3, 1.0, 3.0, 10.0)]
    with Pool(18) as pool: res = pool.map(job, jobs)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth_transplant.json'), 'w'), indent=1)
    for o in res:
        f, a, b = (np.array(o[k])[:, 0].mean() for k in ('full', 'drop0', 'drop5'))
        print('%-4s r %4d m %4.1f | slow EaR full %.3f  drop0 %.3f  drop5 %.3f (truth 0.465) | max swing %.3f' % (o['where'], o['r'], o['m'], f, a, b, max(abs(a - f), abs(b - f))), flush=True)
