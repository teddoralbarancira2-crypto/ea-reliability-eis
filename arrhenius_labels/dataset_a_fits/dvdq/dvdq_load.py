# -*- coding: utf-8 -*-
"""dV/dQ ground-truth attempt, step 1: extract the 0.5C CC charge and discharge curves of every dataset A capacity
test (2 cells x checkpoints 40..480 x 7 T, 1 s logging), and cache Q(V) curves. Protocol per file: rest, discharge to
2.5 V, rest, CC charge to 4.2 V (no CV), rest, CC discharge to 2.5 V, rest. Writes curves.npz. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..'))
import config
import numpy as np, os, re, glob
from multiprocessing import Pool
SRC = os.path.join(config.DATA_A_RAW, 'Capacity test')
HERE = os.path.dirname(os.path.abspath(__file__))

def segs(path):
    a = np.loadtxt(path, skiprows=1); t, V, I = a[:, 0], a[:, 1], a[:, 2]
    cut = np.r_[0, np.where(np.abs(np.diff(I)) > 50)[0] + 1, len(I)]; out = {}
    for s, e in zip(cut[:-1], cut[1:]):
        if e - s < 600: continue
        Im = np.median(I[s:e])
        if abs(Im) < 500: continue
        Q = np.r_[0, np.cumsum(0.5 * (I[s + 1:e] + I[s:e - 1]) * np.diff(t[s:e]))] / 3.6e6   # Ah
        key = 'chg' if Im > 0 else 'dis'
        if key not in out or e - s > len(out[key][0]): out[key] = (np.abs(Q), V[s:e], float(Im))
    return out

def job(path):
    m = re.search(r'Cell(\d)_tem(\d\d)_CAP_(\d{3})', path); c, T, n = int(m[1]), int(m[2]), int(m[3])
    try: return (c, T, n, segs(path))
    except Exception as ex: return (c, T, n, str(ex))

if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(SRC, '*_CAP_*.txt')))
    with Pool(16) as pool: res = pool.map(job, files)
    store = {}
    for c, T, n, s in res:
        if isinstance(s, str): print('FAIL', c, T, n, s); continue
        for k, (Q, V, I) in s.items():
            store['c%d_T%02d_n%03d_%s_Q' % (c, T, n, k)] = Q; store['c%d_T%02d_n%03d_%s_V' % (c, T, n, k)] = V
        print('cell %d T %2d cyc %3d | dis %.3f Ah (I %.0f mA) | chg %.3f Ah' % (c, T, n, s['dis'][0][-1] if 'dis' in s else np.nan,
              s['dis'][2] if 'dis' in s else np.nan, s['chg'][0][-1] if 'chg' in s else np.nan))
    np.savez_compressed(os.path.join(HERE, 'curves.npz'), **store)
