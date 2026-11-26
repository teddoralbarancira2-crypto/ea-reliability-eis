# -*- coding: utf-8 -*-
"""Drop-one-temperature jackknife of the joint fit at every checkpoint, warm-started from joint_age_s{S}.json.
Also refits with residuals scaled per temperature (polarization span). Writes jk_age_s{S}.json. py -3.11 jk_age.py S"""
import numpy as np, json, sys
from multiprocessing import Pool
from common import load, ok, TS, CYC, JS
import joint_age as JA
S = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FULL = json.load(open('joint_age_s%d.json' % S))

def job(a):
    c, j = a; rec = [r for r in FULL['cell%d' % (c + 1)] if r['cyc'] == CYC[j]][0]; q = np.array(rec['q'])
    ks = [k for k in range(7) if ok(c, j, k)]; data = [load(c, j, k, S) for k in ks]
    xs = 1 / (JA.kB * (TS[ks] + 273.15)) - 1 / (JA.kB * JA.TREF); out = dict(cyc=int(CYC[j]), jk=[])
    for d in range(len(ks)):
        keep = [i for i in range(len(ks)) if i != d]
        r = JA.fit([data[i] for i in keep], xs[keep], q); out['jk'].append(dict(drop=int(TS[ks[d]]), proc=JA.procs(r, len(keep))[0]))
    span = np.array([d[1].max() - d[1].min() for d in data]); sc = span / span.mean()
    dw = [(f, zr / s_, zi / s_) for (f, zr, zi), s_ in zip(data, sc)]
    _res = JA.resid
    JA.resid = lambda p, dd, xx: _res_w(p, dd, xx, sc, _res)
    r = JA.fit(data, xs, q); JA.resid = _res; out['weighted'] = JA.procs(r, len(ks))[0]
    return (c, int(CYC[j])), out

def _res_w(p, dd, xx, sc, f0):
    r = f0(p, dd, xx); n = len(dd[0][0]); w = np.repeat(sc, 2 * n); return r / w

if __name__ == '__main__':
    with Pool(8) as pool: res = pool.map(job, [(c, j) for c in (0, 1) for j in JS])
    js = {'cell%d_%d' % (c + 1, n): v for (c, n), v in res}
    json.dump(js, open('jk_age_s%d.json' % S, 'w'), indent=1)
    for key, v in js.items():
        sl = [p['proc'][2] for p in v['jk']]
        print('%-10s slow R25 %s | EaR %s | Eat %s | weighted EaR %.2f Eat %.2f R %.2f' % (key, ' '.join('%.1f' % p['R25'] for p in sl),
              ' '.join('%.2f' % p['Ea'] for p in sl), ' '.join('%.2f' % p['Et'] for p in sl), v['weighted'][2]['Ea'], v['weighted'][2]['Et'], v['weighted'][2]['R25']))
