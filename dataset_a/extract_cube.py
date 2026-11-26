# -*- coding: utf-8 -*-
"""Extract the dataset A 50E EIS cube: cell x cycle x T x SOC-step x freq, plus capacity per T.
Out: cube.npz (Z complex [2,12,7,21,50], V [2,12,7,21], f [50]), cap.json. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import os, re, json, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
D = config.DATA_A_RAW
OUT = config.WORK
CYC = list(range(40, 481, 40))
F0 = 10 ** np.linspace(np.log10(10001.0), np.log10(0.10011718), 50); TS = [0, 5, 15, 25, 35, 45, 55]
def eis(path):
    d = pd.read_csv(path, sep='\t', encoding='latin1', usecols=[0, 1, 2, 3, 4, 5])
    d.columns = ['t', 'V', 'I', 'f', 're', 'im']; e = d[d.f > 0].reset_index(drop=True)
    br = np.where(np.diff(e.f.values) > 0)[0] + 1; segs = np.split(np.arange(len(e)), br)
    out = []
    for s in segs:
        x = e.iloc[s]
        if len(x) < 40: continue
        out.append((x.f.values, x.re.values - 1j * x.im.values, x.V.mean(), x.t.iloc[0]))
    return out
def cap(path):
    d = pd.read_csv(path, sep='\t', encoding='latin1', usecols=[0, 1, 2]); d.columns = ['t', 'V', 'I']
    dt = np.diff(d.t.values, prepend=d.t.values[0]); q = d.I.values * dt / 3600
    # largest contiguous discharge block
    dis = d.I.values < -100; best = 0; acc = 0
    for k in range(len(q)):
        if dis[k]: acc += -q[k]
        else: best = max(best, acc); acc = 0
    best = max(best, acc)
    return float(best), float(d.I.values[dis].mean()) if dis.any() else 0.0
def job(a):
    c, n, T = a
    p = os.path.join(D, 'OCV-EIS test', (config.DATA_A_PREFIX + 'Cell%d_tem%02d_EIS_%03d_CA%d.txt') % (c, T, n, c))
    q = os.path.join(D, 'Capacity test', (config.DATA_A_PREFIX + 'Cell%d_tem%02d_CAP_%03d_CA%d.txt') % (c, T, n, c))
    r = {'key': a, 'eis': None, 'cap': None}
    try: r['eis'] = eis(p)
    except Exception as ex: r['err'] = str(ex)
    try: r['cap'] = cap(q)
    except Exception as ex: r['cerr'] = str(ex)
    return r
if __name__ == '__main__':
    jobs = [(c, n, T) for c in (1, 2) for n in CYC for T in TS]
    Z = np.full((2, 12, 7, 21, 50), np.nan + 0j); V = np.full((2, 12, 7, 21), np.nan); F = None; CAP = {}; LOG = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(job, jobs):
            c, n, T = r['key']; i, j, k = c - 1, CYC.index(n), TS.index(T)
            sp = r['eis'] or []
            LOG.append('%s nspec=%d %s %s' % (r['key'], len(sp), r.get('err', ''), r.get('cerr', '')))
            for s, (f, z, v, t0) in enumerate(sp[:21]):
                if len(f) == 50 and F is None: F = f
                if len(f) != 50:   # 51-point grid on later checkpoints: resample onto the 50-point grid in log f
                    o = np.argsort(f); lf = np.log10(f[o]); g = np.log10(F0)
                    z = np.interp(g, lf, z[o].real) + 1j * np.interp(g, lf, z[o].imag)
                Z[i, j, k, s] = z; V[i, j, k, s] = v
            CAP['%d_%d_%d' % (c, n, T)] = r['cap']
            print(LOG[-1], r['cap'], flush=True)
    np.savez(os.path.join(OUT, 'cube.npz'), Z=Z, V=V, f=F, cyc=CYC, T=TS)
    json.dump(CAP, open(os.path.join(OUT, 'cap.json'), 'w'), indent=1); open(os.path.join(OUT, 'extract_log.txt'), 'w').write('\n'.join(LOG))
