# -*- coding: utf-8 -*-
"""Stage 1: DRT of every HPPC Rest spectrum (2 cells x 10 T x 6 SOC, 10 kHz-0.1 Hz) on ONE shared tau grid with ONE fixed lambda.
Model (f <= 2 kHz) Z = R0 + jwL + sum g_j dlnτ/(1+jwτ_j), NNLS, all 51 points (inductive ones handled by L).
Writes drt_all.npz (gamma cube) and a quick look figure. py -3.11"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import config
import numpy as np, scipy.io, os
from scipy.optimize import nnls
HERE = os.path.dirname(os.path.abspath(__file__))
DAT = config.DATA_B_DIR
CELLS = ['no1_ch3', 'no2_ch4']; TEMPS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
REST = {100: 0, 90: 1, 70: 8, 50: 15, 30: 22, 10: 29}          # column-block index of each Rest spectrum
SOCS = [100, 90, 70, 50, 30, 10]
FMAX = 2000.0                                                   # above ~2 kHz the lead inductance is not a pure L
TAU = np.logspace(np.log10(1 / (2 * np.pi * FMAX) * 0.3), np.log10(1 / (2 * np.pi * 0.1) * 3), 140)
DL = np.log(TAU[1] / TAU[0])

def load(cell, t, soc):
    d = scipy.io.loadmat(os.path.join(DAT, '%s_all_rearranged.mat' % cell))['%s_soc_crate_%dt_rearranged' % (cell, t)]
    i = REST[soc]; f, zr, zi = d[:, 3 * i], d[:, 3 * i + 1], d[:, 3 * i + 2]; o = np.argsort(f)[::-1]
    f, zr, zi = f[o], zr[o], zi[o]; m = f <= FMAX * 1.001
    return f[m], zr[m], zi[m]          # zi = -Im Z (capacitive > 0)

def design(f):
    w = 2 * np.pi * f; wt = w[:, None] * TAU[None, :]
    Ar = DL / (1 + wt ** 2); Ai = DL * wt / (1 + wt ** 2)          # real part, and -Im part
    n = len(f); one = np.ones(n)
    A = np.block([[one[:, None], np.zeros((n, 1)), Ar], [np.zeros((n, 1)), -w[:, None], Ai]])
    return A

def drt(f, zr, zi, lam):
    A = design(f); D = np.diff(np.eye(len(TAU)), 2, axis=0)
    D = np.hstack([np.zeros((D.shape[0], 2)), D])
    s = 1e3                                                       # work in mOhm
    x, _ = nnls(np.vstack([A, np.sqrt(lam) * D]), np.concatenate([zr * s, zi * s, np.zeros(D.shape[0])]))
    fit = A @ x; res = np.concatenate([zr * s, zi * s]) - fit
    return x[0], x[1] / s, x[2:], np.sqrt(np.mean(res ** 2))      # R0 mOhm, L H, gamma mOhm, rms mOhm

if __name__ == '__main__':
    import sys
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    G = np.zeros((2, 10, 6, len(TAU))); R0 = np.zeros((2, 10, 6)); L = np.zeros_like(R0); RMS = np.zeros_like(R0)
    for c, cell in enumerate(CELLS):
        for k, t in enumerate(TEMPS):
            for s, soc in enumerate(SOCS):
                f, zr, zi = load(cell, t, soc)
                R0[c, k, s], L[c, k, s], G[c, k, s], RMS[c, k, s] = drt(f, zr, zi, lam)
    np.savez(os.path.join(HERE, 'drt_all.npz'), G=G, R0=R0, L=L, RMS=RMS, TAU=TAU, TEMPS=TEMPS, SOCS=SOCS, lam=lam)
    print('lam', lam, 'rms mOhm median %.4f max %.4f' % (np.median(RMS), RMS.max()), 'L nH median %.1f' % (1e9 * np.median(L)))
    print('R0 mOhm by T (cell1, SOC50):', np.round(R0[0, :, 3], 2))
