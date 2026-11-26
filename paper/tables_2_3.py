# -*- coding: utf-8 -*-
"""Tables 2 and 3 of the paper from the stored benchmark results.
Table 2 (fixed aging state, ident/consequence2.json): for each held-out spectrum the error is the largest |relative error| of Re Z over 10, 1
and 0.1 Hz; median and maximum over all held-out spectra of a split. Table 3 (calibrate once, ident/aging_transfer.json): the same error at 25 C,
median and 90th percentile over cases. py -3.11"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import config
ID = os.path.join(config.AL, 'ident')
R = json.load(open(os.path.join(ID, 'consequence2.json')))
M = (('rel', 'process model'), ('lut_rel', 'lookup table'), ('base_rel', 'Arrhenius per f'), ('lin_rel', 'linear per f'))
print('Table 2: median (max) over held-out spectra, %')
for ds, nm in ((('A ', 'Auto'), 'A'), (('B ', 'L' + 'GES'), 'B')):   # new and old case labels
    for sp in ('interp', 'cold', 'warm'):
        C = [o for o in R if o['case'].startswith(ds) and o['split'] == sp]; row = []
        for k, _ in M:
            e = np.concatenate([np.max(np.abs([o['%s_%g' % (k, f)] for f in (10, 1, 0.1)]), axis=0) * 100 for o in C])
            row.append('%.1f (%.1f)' % (np.median(e), e.max()))
        print('  %s %-6s (%2d spectra): %s' % (nm, sp, len(e), ' | '.join(row)))
T = json.load(open(os.path.join(ID, 'aging_transfer.json')))
print('Table 3: median / 90th percentile over cases, %')
for A in (40, 240):
    s = [r for r in T if r['A'] == A]
    print('  calibrated at cycle %d (%d cases): %s' % (A, len(s), ' | '.join(
        '%.1f / %.1f' % (np.median(e), np.percentile(e, 90)) for e in ([max(abs(r[m][k]) for k in r[m]) * 100 for r in s] for m in ('ours', 'lut', 'arr', 'lin')))))
