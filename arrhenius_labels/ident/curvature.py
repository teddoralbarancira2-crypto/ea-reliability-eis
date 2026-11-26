# -*- coding: utf-8 -*-
"""Is the dataset B slow-process Arrhenius plot curved? K=2 joint fits (tailT variant: CPE tail free per T) on sliding
temperature windows. If the slow Ea_R depends on the window (low when cold, high when warm) the 0 C 'swing' is curvature,
i.e. a temperature-dependent Ea, not noise or proximity. Also fast Ea_R for comparison. Writes curvature.json. py -3.11"""
import numpy as np, json, os
from multiprocessing import Pool
from cold_split import getdata, XS, TT, CELLS
from cold_variants import fit
WIN = [(0, 20), (5, 25), (10, 30), (15, 35), (20, 40), (25, 45), (0, 45), (10, 45)]
def job(a):
    cell, soc, (t0, t1) = a; w, data = getdata(cell, soc); keep = (TT >= t0) & (TT <= t1)
    r = fit([d for d, k in zip(data, keep) if k], XS[keep], w, 'tailT'); return '%s_%d_%d-%d' % (cell, soc, t0, t1), dict(slow=r['slowEaR'], fast=r['fastEaR'], slowR=r['slowR'], rms=r['rms'])
if __name__ == '__main__':
    jobs = [(c, s, wn) for c in CELLS for s in (50, 70) for wn in WIN]
    with Pool(16) as pool: R = dict(pool.map(job, jobs))
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'curvature.json'), 'w'), indent=1)
    for c in CELLS:
        for s in (50, 70):
            print('%s SOC%d ' % (c, s) + ' | '.join('%d-%dC slow %.2f fast %.2f' % (a, b, R['%s_%d_%d-%d' % (c, s, a, b)]['slow'], R['%s_%d_%d-%d' % (c, s, a, b)]['fast']) for a, b in WIN))
