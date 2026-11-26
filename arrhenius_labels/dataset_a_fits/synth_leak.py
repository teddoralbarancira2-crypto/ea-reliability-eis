# -*- coding: utf-8 -*-
"""Synthetic leakage test. Ground truth: ONE fast ZARC with a fixed label (R25 5.5 mOhm, Ea_R 0.45, tau25 0.5 ms, Ea_tau 0.45,
alpha 0.62) + a growing slow arc with a fixed label (Ea_R = Ea_tau = 0.53, alpha 0.88, tau25 proportional to R25 as measured)
whose shape is Havriliak-Negami with beta = 1 (symmetric ZARC, control), 0.7 or 0.5 (asymmetric). R0(T), lead and CPE tail
are taken from the variant-B fit of cell 1 cycle 240. Gaussian noise 0.04 mOhm. Each synthetic set is fitted with the exact
variant-B pipeline (3 ZARCs in tau windows, symmetric slow arc). If the fitted fast-band Ea_eff falls as the slow arc grows
while the truth is fixed at 0.45, the measured decline can be leakage. Writes synth_leak.json. py -3.11"""
import numpy as np, json
from multiprocessing import Pool
from common import Z, F, TS, CYC
import joint_hf as JH
TT = TS.astype(float); STB = JH.Setup('B'); STE = JH.Setup('E'); f = F[STB.MF]
xs = 1 / (JH.kB * (TT + 273.15)) - 1 / (JH.kB * JH.TREF)
SLOW = [1.5, 3.0, 5.0, 7.0, 9.5, 12.0]; BETAS = [1.0, 0.7, 0.5]

def base():
    j = CYC.index(240); data = [(f, Z[0, j, k, 10][STB.MF].real * 1e3, -Z[0, j, k, 10][STB.MF].imag * 1e3) for k in range(7)]
    return JH.fit_data(data, TT, STB, seed=1)

def synth(args):
    xb, Rs, beta, seed = args
    nh = len(xb) - (5 * JH.K + 3); head = xb[:nh]; tail = xb[nh + 5 * JH.K:nh + 5 * JH.K + 3]
    tau_s = 0.125 * Rs / 12.0
    q = [np.log(5.5), 0.45, np.log(5e-4), 0.45, 0.62,  np.log(1e-3), 0.3, np.log(5e-3), 0.3, 0.9,
         np.log(Rs), 0.53, np.log(tau_s), 0.53, 0.88] + list(tail) + [beta]
    zs = JH.model(np.array(list(head) + q), f, xs, TT, STE); rng = np.random.default_rng(seed)
    data = [(f, z.real + rng.normal(0, 0.04, len(f)), -z.imag + rng.normal(0, 0.04, len(f))) for z in zs]
    r = JH.fit_data(data, TT, STB, seed=seed); qq = r['q']; e, fb25 = JH.fast_band_ea(qq)
    return dict(Rs=Rs, beta=beta, rms=r['rms'], fb_Ea=float(e), fb_R25=float(fb25), slow_R25=float(np.exp(qq[10])), slow_EaR=qq[11], slow_Eat=qq[13],
                mid_R25=float(np.exp(qq[5])), mid_EaR=qq[6], mid_Eat=qq[8], fast_EaR=qq[1])

if __name__ == '__main__':
    b = base(); xb = np.array(b['x']); print('base fit rms %.3f' % b['rms'], flush=True)
    jobs = [(xb, Rs, beta, 100 * i + k) for i, beta in enumerate(BETAS) for k, Rs in enumerate(SLOW)]
    with Pool(8) as pool: res = pool.map(synth, jobs)
    json.dump(res, open('synth_leak.json', 'w'), indent=1)
    for r in res:
        print('beta %.1f slow R25 %5.1f | fit rms %.3f | fast-band Ea_eff %.3f (truth 0.450) R25 %.2f (truth 5.50) | slow R25 %.2f EaR %.3f Eat %.3f (truth 0.53) | mid R25 %.2f EaR %.2f Eat %.2f' % (
            r['beta'], r['Rs'], r['rms'], r['fb_Ea'], r['fb_R25'], r['slow_R25'], r['slow_EaR'], r['slow_Eat'], r['mid_R25'], r['mid_EaR'], r['mid_Eat']))
