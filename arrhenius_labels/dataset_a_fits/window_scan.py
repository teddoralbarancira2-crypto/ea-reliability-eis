# -*- coding: utf-8 -*-
"""Sliding temperature-window scan of the slow-arc label (variant B, 3.73 V), the same diagnostic that exposed the dataset B
slow process (window-dependent Ea 0.44-0.73 eV). If the dataset A slow arc is genuinely Arrhenius, every window should
return the same Ea_R within noise. Cycles 120..480, both cells. Writes window_scan.json. py -3.11"""
import numpy as np, json
from multiprocessing import Pool
from common import Z, F, TS, CYC, ok
import joint_hf as JH
import sys
WIN = [(0, 25), (5, 35), (15, 45), (25, 55), (0, 55)]; S = int(sys.argv[1]) if len(sys.argv) > 1 else 10
def job(a):
    c, n, (t0, t1) = a; j = CYC.index(n); st = JH.Setup('B')
    ks = [k for k in range(7) if ok(c, j, k) and t0 <= TS[k] <= t1]; TT = TS[ks].astype(float)
    data = [(F[st.MF], Z[c, j, k, S][st.MF].real * 1e3, -Z[c, j, k, S][st.MF].imag * 1e3) for k in ks]
    r = JH.fit_data(data, TT, st, seed=11 * c + j, n_rand=4); q = r['q']
    return '%d_%d_%d-%d' % (c + 1, n, t0, t1), dict(EaR=q[11], Eat=q[13], R25=float(np.exp(q[10])), nT=len(ks), rms=r['rms'])
if __name__ == '__main__':
    jobs = [(c, n, w) for c in (0, 1) for n in (120, 160, 200, 240, 280, 320, 360, 480) for w in WIN]
    with Pool(16) as pool: R = dict(pool.map(job, jobs))
    json.dump(R, open('window_scan_s%d.json' % S, 'w'), indent=1)
    for c in (1, 2):
        for n in (120, 160, 200, 240, 280, 320, 360, 480):
            v = [R['%d_%d_%d-%d' % (c, n, a, b)] for a, b in WIN]; e = [x['EaR'] for x in v]
            print('cell%d cyc%3d R25 %5.2f | ' % (c, n, v[-1]['R25']) + ' | '.join('%d-%dC %.2f (%dT)' % (a, b, x['EaR'], x['nT']) for (a, b), x in zip(WIN, v)) + ' | span %.3f' % (max(e) - min(e)))
