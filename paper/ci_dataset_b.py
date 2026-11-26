# -*- coding: utf-8 -*-
"""Dataset B: Jacobian 95 % CI of the slow and fast Ea_R from the full 10-T joint K=2 fit (restart from joint_ms_K2.json x).
Writes ci_dataset_b.json. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import os, sys, json, numpy as np
AL = config.AL; sys.path.insert(0, AL); os.chdir(AL)
from scipy.optimize import least_squares
from drt_all import load, TEMPS
import joint_fit as J
MS = json.load(open('joint_ms_K2.json')); nT = len(TEMPS); out = {}
for case in ('no1_ch3_50', 'no1_ch3_70', 'no2_ch4_50', 'no2_ch4_70'):
    cell, soc = case.rsplit('_', 1); x0 = np.array(MS[case]['x'])
    data = [(f, zr * 1e3, zi * 1e3) for f, zr, zi in (load(cell, t, int(soc)) for t in TEMPS)]
    r = least_squares(J.resid_joint, x0, args=(data, 2), x_scale='jac', max_nfev=4000)
    s2 = (r.fun ** 2).sum() / (len(r.fun) - len(r.x)); se = np.sqrt(np.diag(np.linalg.pinv(r.jac.T @ r.jac) * s2))
    q = r.x[2 * nT:]; sq = se[2 * nT:]; ks = sorted((0, 5), key=lambda b: q[b + 2])     # fast, slow by tau25
    d = {nm: dict(Ea=float(q[b + 1]), se=float(sq[b + 1]), R25=float(np.exp(q[b]))) for nm, b in zip(('fast', 'slow'), ks)}
    d['rms'] = float(np.sqrt(np.mean(r.fun ** 2))); out[case] = d
    print(case, 'slow %.3f +- %.3f (CI %.3f-%.3f) fast %.3f +- %.3f rms %.4f R25 slow %.3f ratio %.4f' % (d['slow']['Ea'], d['slow']['se'], d['slow']['Ea'] - 1.96 * d['slow']['se'],
          d['slow']['Ea'] + 1.96 * d['slow']['se'], d['fast']['Ea'], d['fast']['se'], d['rms'], d['slow']['R25'], d['rms'] / d['slow']['R25']), flush=True)
json.dump(out, open(os.path.join(config.PAPER, 'ci_dataset_b.json'), 'w'), indent=1)
