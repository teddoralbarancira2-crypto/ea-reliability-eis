# -*- coding: utf-8 -*-
"""dataset A 50E loader + DRT for the Arrhenius-label test across aging. Z cube from dataset_a/extract_cube.py (cube.npz):
[cell 2, checkpoint 12 (cyc 40..480, 400 empty), T 7 (0..55 C), SOC step 21, f 50 (10 kHz..0.1 Hz)]. f <= 2 kHz used."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import config
import numpy as np, os
from scipy.optimize import nnls
CUBE = config.CUBE
_z = np.load(CUBE); Z = _z['Z']; F = _z['f']; TS = np.array(_z['T']); CYC = list(_z['cyc'])
FMAX = 2000.0; MF = F <= FMAX * 1.001
JS = [j for j, n in enumerate(CYC) if n != 400]
def ok(c, j, k):                       # cycle 280 at 5 C is a faulty spectrum
    return not (CYC[j] == 280 and TS[k] == 5) and np.isfinite(Z[c, j, k, 0, 0])
def load(c, j, k, s):
    z = Z[c, j, k, s][MF]; return F[MF], z.real * 1e3, -z.imag * 1e3      # mOhm, zi = -Im
TAU = np.logspace(np.log10(1 / (2 * np.pi * FMAX) * 0.3), np.log10(1 / (2 * np.pi * 0.1) * 3), 140); DL = np.log(TAU[1] / TAU[0])
def drt(f, zr, zi, lam=0.1):
    w = 2 * np.pi * f; wt = w[:, None] * TAU[None, :]; n = len(f)
    A = np.block([[np.ones((n, 1)), np.zeros((n, 1)), DL / (1 + wt ** 2)], [np.zeros((n, 1)), -w[:, None] * 1e-3, DL * wt / (1 + wt ** 2)]])
    D = np.hstack([np.zeros((len(TAU) - 2, 2)), np.diff(np.eye(len(TAU)), 2, axis=0)])
    x, _ = nnls(np.vstack([A, np.sqrt(lam) * D]), np.concatenate([zr, zi, np.zeros(len(TAU) - 2)]))
    return x[0], x[2:], np.sqrt(np.mean((np.concatenate([zr, zi]) - A @ x) ** 2))
