# -*- coding: utf-8 -*-
"""Split-independent fast-band Ea: from each checkpoint's K=3 fit, R_fb(T) = R_fast(T) + R_mid(T) on the measured temperatures,
Arrhenius slope of ln R_fb vs 1/kT. Same for the slow arc and for the whole polarization R_fb + R_slow. py -3.11 fastband.py FILE"""
import numpy as np, json, sys
kB = 8.617333e-5; TREF = 298.15; T = np.array([0, 5, 15, 25, 35, 45, 55]) + 273.15; X = 1 / (kB * T) - 1 / (kB * TREF); x = 1 / (kB * T)
J = json.load(open(sys.argv[1]))
def R(q, k): return np.exp(q[5 * k] + q[5 * k + 1] * X)
rows = []
for key in sorted(J, key=lambda k: (k[:5], int(k.split('_')[1]))):
    q = J[key]['q']; fb = R(q, 0) + R(q, 1); sl = R(q, 2)
    e = lambda y: np.polyfit(x, np.log(y), 1)[0]
    print('%-10s fast-band R25 %.2f Ea_eff %.3f (curv %.3f) | slow R25 %.2f Ea %.3f | total Ea %.3f | fast-band share of R at 0/25/55 C %.2f/%.2f/%.2f' % (
        key, R(q, 0)[3] + R(q, 1)[3], e(fb), np.polyfit(x, np.log(fb), 2)[0], sl[3], e(sl), e(fb + sl), *(fb / (fb + sl))[[0, 3, 6]]))
