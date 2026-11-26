# -*- coding: utf-8 -*-
"""dV/dQ step 3: degradation-mode ground truth by electrode alignment (open-circuit-voltage model fitting).
Pseudo-OCV: the 0.5C CC charge and the following CC discharge share the same top-of-charge state, so both are put on
Q measured from the top and averaged (polarization cancels to first order). Model, Q from top:
    V(Q) = U_p(x_top + Q/C_p) - U_n(y_top - Q/C_n)
with U_p, U_n the MEASURED LG M50 half-cell OCP tables (NMC811, graphite-SiOx; Chen et al. JES 2020, shipped with
PyBaMM) as proxies for the 50E electrodes. Fitted per (cell, T, checkpoint): C_p, C_n (Ah), x_top, y_top. Residual on V
and on dV/dQ (features). Derived: lithium inventory Li = C_p x_top + C_n y_top (Ah of cyclable Li in both
electrodes at the top of charge), LAM_PE = 1 - C_p/C_p(40), LAM_NE = 1 - C_n/C_n(40),
LLI = 1 - Li/Li(40). Fitted independently at 25 and 45 C: a real mode estimate must not depend on test temperature.
Writes dva.json, dva_fit_examples.png. py -3.11"""
import numpy as np, os, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); D = np.load(os.path.join(HERE, 'curves.npz'))
CYC = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480]

def table(fn):
    a = np.loadtxt(os.path.join(HERE, 'ref', fn), delimiter=',', comments='#'); o = np.argsort(a[:, 0]); return a[o, 0], a[o, 1]
def smooth_table(x, u, win=0.03):
    g = np.arange(x[0], x[-1], 0.001); v = np.interp(g, x, u); k = int(win / 0.001) | 1
    return g, savgol_filter(v, k, 2)
XP, UP = smooth_table(*table('nmc_LGM50_ocp_Chen2020.csv')); XN, UN = smooth_table(*table('graphite_LGM50_ocp_Chen2020.csv'))
def up_an(x): return -0.8090 * x + 4.4875 - 0.0428 * np.tanh(18.5138 * (x - 0.5542)) - 17.7326 * np.tanh(15.7890 * (x - 0.3117)) + 17.5842 * np.tanh(15.9308 * (x - 0.3120))
def un_an(y): return 1.9793 * np.exp(-39.3631 * y) + 0.2482 - 0.0909 * np.tanh(29.8538 * (y - 0.1234)) - 0.04478 * np.tanh(14.9159 * (y - 0.2769)) - 0.0205 * np.tanh(30.4444 * (y - 0.6103))
REF = {'table': (lambda x: np.interp(x, XP, UP), lambda y: np.interp(y, XN, UN)), 'analytic': (up_an, un_an)}
Up, Un = REF['table']

def pseudo_ocv(c, T, n, dq=0.005):
    k = 'c%d_T%02d_n%03d_' % (c, T, n)
    if k + 'chg_Q' not in D or k + 'dis_Q' not in D: return None
    Qc, Vc, Qd, Vd = D[k + 'chg_Q'], D[k + 'chg_V'], D[k + 'dis_Q'], D[k + 'dis_V']
    if Qc[-1] < 3 or Qd[-1] < 3: return None
    qc_top = Qc[-1] - Qc                                         # charge, Q from top (descending)
    Qmax = min(Qc[-1], Qd[-1]) - 0.02; g = np.arange(0.02, Qmax, dq)
    vc = np.interp(g, qc_top[::-1], Vc[::-1]); vd = np.interp(g, Qd, Vd)
    cv = vc >= 4.195; eta = 0.5 * (vc - vd); i0 = np.argmax(~cv); eta[cv] = eta[i0]   # CV hold at the top: carry eta
    return g, vd + eta, vc, vd

def model(p, Q, ref='table'):
    Cp, Cn, xt, yt = p; up, un = REF[ref]; return up(xt + Q / Cp) - un(yt - Q / Cn)

def dvdq(Q, V, win=31):
    return savgol_filter(np.gradient(savgol_filter(V, win, 2), Q[1] - Q[0]), win, 2)

def resid(p, Q, V, dV, wd, ref):
    m = model(p, Q, ref); return np.r_[m - V, wd * (dvdq(Q, m) - dV)]

def fit(Q, V, wd=0.3, ref='table'):
    dV = dvdq(Q, V); best = None
    lo, hi = [3.0, 3.0, 0.15, 0.55], [9.0, 9.0, 0.6, 1.0]
    for Cp0 in (5.0, 6.0):
        for Cn0 in (4.8, 5.6):
            for yt0 in (0.8, 0.95):
                r = least_squares(resid, [Cp0, Cn0, 0.27, yt0], args=(Q, V, dV, wd, ref), bounds=(lo, hi), x_scale=[1, 1, 0.1, 0.1], diff_step=1e-4)
                if best is None or r.cost < best.cost: best = r
    J = best.jac; s2 = (best.fun ** 2).sum() / (len(best.fun) - 4)
    se = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * s2))
    rmsV = float(np.sqrt(np.mean((model(best.x, Q, ref) - V) ** 2)))
    return best.x, se, rmsV

VARIANTS = {'base': dict(wd=0.3, ref='table'), 'Vonly': dict(wd=0.0, ref='table'), 'analytic': dict(wd=0.3, ref='analytic')}
def job(a):
    c, T, n, var = a; po = pseudo_ocv(c, T, n)
    if po is None: return None
    Q, V, _, _ = po; p, se, rms = fit(Q, V, **VARIANTS[var]); Cp, Cn, xt, yt = p
    return dict(var=var, cell=c, T=T, cyc=n, Cp=Cp, Cn=Cn, xt=xt, yt=yt, se=list(se), rmsV=rms, Li=Cp * xt + Cn * yt, Qtot=float(Q[-1] + 0.02))

if __name__ == '__main__':
    jobs = [(c, T, n, v) for v in VARIANTS for c in (1, 2) for T in (25, 45) for n in CYC]
    with Pool(16) as pool: res = [r for r in pool.map(job, jobs) if r]
    for v in VARIANTS:
      for c in (1, 2):
        for T in (25, 45):
            rr = [r for r in res if r['cell'] == c and r['T'] == T and r['var'] == v]; b = rr[0]
            for r in rr:
                r['LAM_PE'] = 1 - r['Cp'] / b['Cp']; r['LAM_NE'] = 1 - r['Cn'] / b['Cn']; r['LLI'] = 1 - r['Li'] / b['Li']
                if r['cyc'] in (40, 160, 240, 360, 480): print(v, 'cell %d T %d cyc %3d | Q %.3f | Cp %.3f±%.3f Cn %.3f±%.3f xt %.3f yt %.3f Li %.3f | LAM_PE %5.1f%% LAM_NE %5.1f%% LLI %5.1f%% | rmsV %.1f mV' % (
                    c, T, r['cyc'], r['Qtot'], r['Cp'], r['se'][0], r['Cn'], r['se'][1], r['xt'], r['yt'], r['Li'], 100 * r['LAM_PE'], 100 * r['LAM_NE'], 100 * r['LLI'], 1e3 * r['rmsV']))
    json.dump(res, open(os.path.join(HERE, 'dva.json'), 'w'), indent=1)
    f, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, n in enumerate((40, 480)):
        r = [x for x in res if x['cell'] == 1 and x['T'] == 45 and x['cyc'] == n and x['var'] == 'base'][0]; Q, V, vc, vd = pseudo_ocv(1, 45, n); m = model([r['Cp'], r['Cn'], r['xt'], r['yt']], Q)
        ax[0, j].plot(Q, vc, color='#c26a12', lw=0.7, label='charge 0.5C'); ax[0, j].plot(Q, vd, color='#1f5fa8', lw=0.7, label='discharge 0.5C')
        ax[0, j].plot(Q, V, 'k', lw=1.2, label='pseudo-OCV'); ax[0, j].plot(Q, m, 'r--', lw=1, label='electrode model'); ax[0, j].set_title('cell 1, 45 C, cycle %d' % n); ax[0, j].legend(fontsize=7)
        ax[1, j].plot(Q, -dvdq(Q, V), 'k', lw=1.2, label='pseudo-OCV'); ax[1, j].plot(Q, -dvdq(Q, m), 'r--', lw=1, label='model'); ax[1, j].set_ylim(0, 0.6)
        ax[1, j].set_xlabel('Q from top of charge (Ah)'); ax[1, j].set_ylabel('-dV/dQ (V/Ah)')
    f.tight_layout(); f.savefig(os.path.join(HERE, 'dva_fit_examples.png'), dpi=110)
