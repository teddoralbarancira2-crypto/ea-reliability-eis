# -*- coding: utf-8 -*-
"""Aging transfer ("calibrate once"): temperature compensation calibrated at checkpoint A (all 7 T) and applied at a
later checkpoint B, where only ONE spectrum at T1 is available; target = the measured spectrum at T2 = 25 C at B.
dataset A 3.73 V (S=10), both cells. A in {40, 240}; B = every later checkpoint; T1 in {0, 5, 15, 35, 45, 55}.
Methods (all calibrated on A only):
 ours  process-resolved: K=3 + CPE + lead model fitted to the single B spectrum at T1 (time constants and widths held
       near A's values by a soft prior), each element then moved to T2 with A's activation energies
       (Ea_R, Ea_tau per process, tail Eq, R0 slope)
 lut   per-frequency ln Re Z shape vs 1/T from A (piecewise), applied as a multiplicative shift
 arr   per-frequency Arrhenius (Schinagl 2025), Ea(f) by least squares over A's 7 temperatures
 lin   per-frequency linear (Schinagl 2025), slope k(f) in mOhm/K from A
Metric: relative error of Re Z at 10, 1, 0.1 Hz at 25 C. Writes aging_transfer.json. py -3.11"""
import numpy as np, json, os, sys
from multiprocessing import Pool
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); AUTO = os.path.join(os.path.dirname(HERE), 'dataset_a_fits'); sys.path.insert(0, AUTO)
os.chdir(AUTO)
from common import Z, F, TS, CYC, ok
import joint_hf as JH
kB = 8.617333e-5; TREF = 298.15; S = 10; FQ = (10.0, 1.0, 0.1); K = 3
ST = JH.Setup('B'); MF = ST.MF; f = F[MF]; w = 2 * np.pi * f
HF = json.load(open('hf_B_s10.json'))
def X(T): return 1 / (kB * (np.asarray(T, float) + 273.15)) - 1 / (kB * TREF)
def ki(T): return list(TS).index(T)
def spec(c, n, T): return Z[c, CYC.index(n), ki(T), S][MF] * 1e3
def at(z, fq): return z.real[np.argmin(np.abs(f - fq))]

def calib(c, A):
    v = HF['cell%d_%d' % (c + 1, A)]; q = np.array(v['q']); T_ = np.array(v['temps'], float); R0 = np.array(v['R0'])
    sl, ic = np.polyfit(T_ - 25, R0, 1)                      # R0(T) = a (1 + b (T - 25)) -> b = slope / a
    cal = dict(R0b=sl / ic, L1=3e-6, L2=float(v['L2']), RL=float(v['RL']), q=q)   # L1 is refitted per spectrum
    temps = v['temps']; Zs = {T: spec(c, A, T) for T in temps}
    cal['temps'] = temps; cal['Zs'] = Zs; return cal

def zmodel(p, cal):
    R0, lL1, lL2, lRL = p[:4]; z = R0 + 1j * w * np.exp(lL1) + (1j * w * np.exp(lL2) * np.exp(lRL)) / (np.exp(lRL) + 1j * w * np.exp(lL2))
    for k in range(K):
        lR, lt, a = p[4 + 3 * k:7 + 3 * k]; z = z + np.exp(lR) / (1 + (1j * w * np.exp(lt)) ** a)
    lQ, nn = p[4 + 3 * K:6 + 3 * K]; return z + 1 / (np.exp(lQ) * (1j * w) ** nn)

def ours(cal, zB, T1, T2):
    q = cal['q']; x1, x2 = X(T1), X(T2)
    p0 = [zB.real[np.argmin(np.abs(f - 1500))] * 0.97, np.log(cal['L1']), np.log(cal['L2']), np.log(cal['RL'])]
    pri = []
    for k in range(K):
        lR, Ea, lt, Et, a = q[5 * k:5 * k + 5]; p0 += [lR + Ea * x1, lt + Et * x1, a]; pri.append((lt + Et * x1, a))
    p0 += [q[15] - q[16] * x1, q[17]]; p0 = np.array(p0)
    lo = [0, np.log(1e-9), np.log(1e-9), np.log(0.01)]; hi = [100, np.log(1e-3), np.log(1e-3), np.log(1e3)]
    for k in range(K): lo += [np.log(1e-3), np.log(1e-7), 0.4]; hi += [np.log(500), np.log(20.0), 1.0]
    lo += [np.log(1e-3), 0.3]; hi += [np.log(1e6), 1.0]; lo, hi = np.array(lo), np.array(hi)
    def res(p):
        e = zmodel(p, cal) - zB; r = [e.real, e.imag]
        r.append(np.array([0.05 * (p[5 + 3 * k] - pri[k][0]) / 1.0 for k in range(K)] + [0.05 * (p[6 + 3 * k] - pri[k][1]) / 0.1 for k in range(K)]))
        return np.concatenate(r)
    r = least_squares(res, np.clip(p0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), x_scale='jac', max_nfev=6000); p = r.x.copy()
    b = cal['R0b']; p[0] = p[0] * (1 + b * (T2 - 25)) / (1 + b * (T1 - 25))
    for k in range(K):
        p[4 + 3 * k] += q[5 * k + 1] * (x2 - x1); p[5 + 3 * k] += q[5 * k + 3] * (x2 - x1)
    p[4 + 3 * K] -= q[16] * (x2 - x1)
    return zmodel(p, cal), float(np.sqrt(np.mean(r.fun[:2 * len(w)] ** 2)))

def perf(cal, zB, T1, T2, method):
    temps = np.array(cal['temps'], float); out = {}
    for fq in FQ:
        y = np.array([at(cal['Zs'][T], fq) for T in cal['temps']]); y1 = at(zB, fq)
        if method == 'arr':
            Ea = np.polyfit(X(temps), np.log(y), 1)[0]; out[fq] = y1 * np.exp(Ea * (X(T2) - X(T1)))
        elif method == 'lin':
            k = np.polyfit(temps, y, 1)[0]; out[fq] = y1 + k * (T2 - T1)
        else:
            o = np.argsort(X(temps)); xx, ly = X(temps)[o], np.log(y)[o]; g = lambda v: np.interp(v, xx, ly)
            out[fq] = y1 * np.exp(g(X(T2)) - g(X(T1)))
    return out

def job(a):
    c, A = a; cal = calib(c, A); rows = []
    for B in [n for n in CYC if n > A and n != 400]:
        jB = CYC.index(B)
        if not ok(c, jB, ki(25)): continue
        z25 = spec(c, B, 25)
        for T1 in (0, 5, 15, 35, 45, 55):
            if not ok(c, jB, ki(T1)): continue
            zB = spec(c, B, T1); zo, rms = ours(cal, zB, T1, 25); row = dict(cell=c + 1, A=int(A), B=int(B), T1=int(T1), fit_rms=rms)
            row['ours'] = {str(fq): float((at(zo, fq) - at(z25, fq)) / at(z25, fq)) for fq in FQ}
            for m in ('lut', 'arr', 'lin'):
                pr = perf(cal, zB, T1, 25, m); row[m] = {str(fq): float((pr[fq] - at(z25, fq)) / at(z25, fq)) for fq in FQ}
            rows.append(row)
    return rows

if __name__ == '__main__':
    with Pool(4) as pool: R = sum(pool.map(job, [(c, A) for c in (0, 1) for A in (40, 240)]), [])
    json.dump(R, open(os.path.join(HERE, 'aging_transfer.json'), 'w'), indent=1)
    for A in (40, 240):
        print('== calibrated at cycle %d (applied at later checkpoints, both cells)' % A)
        for T1 in (0, 5, 15, 35, 45, 55):
            s = [r for r in R if r['A'] == A and r['T1'] == T1]
            line = '  from %2d C to 25 C (n=%2d): ' % (T1, len(s))
            for m in ('ours', 'lut', 'arr', 'lin'):
                e = [max(abs(r[m][str(fq)]) for fq in FQ) * 100 for r in s]
                line += '%s med %.1f max %.1f | ' % (m, np.median(e), max(e))
            print(line, flush=True)
        s = [r for r in R if r['A'] == A]
        print('  ALL: ' + ' | '.join('%s med %.1f p90 %.1f' % (m, np.median([max(abs(r[m][str(fq)]) for fq in FQ) * 100 for r in s]), np.percentile([max(abs(r[m][str(fq)]) for fq in FQ) * 100 for r in s], 90)) for m in ('ours', 'lut', 'arr', 'lin')))
        print('  single-spectrum fit rms (mOhm): median %.3f max %.3f' % (np.median([r['fit_rms'] for r in s]), max(r['fit_rms'] for r in s)))
