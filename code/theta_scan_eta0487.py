"""Section 5: no theta on a grid of step 0.25 in [8, 24] verifies the criterion at eta = 0.487 (4000 cells)."""
import numpy as np, cert_interval as ci
best = None; n = 0
for th in np.arange(8, 24.01, 0.25):
    r = ci.sharp_kappa(1, 2.25, '0.05', '0.487', str(round(th, 2)), nt=4000)
    print(round(th, 2), r['kappa'], bool(r['certified'])); k = float(r['kappa'].lower()); n += bool(r['certified'])
    if best is None or k < best[0]: best = (k, th)
print('min lower bound', best, 'certified count', n)
