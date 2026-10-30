"""Arb part of calib_season1.py, recomputed from the saved calibration (q0, zeta) so that the raw iPinYou file
is not needed. Writes calib_season1_cert_fresh.json."""
import json, numpy as np, cert_general as cg
S = json.load(open('calib_season1.json')); out = {}
for top in (450, 600):
    res = S[f'top={top}']; mg = np.array([0, 150, 300, top]) / 3000; q0 = np.array(res['q0']); z = res['zeta_fit']; cert = {}
    for zeta in (round(z, 4), 0.1, 0.2):
        for eta in ('0.3', '0.45'):
            key = f'zeta={zeta},eta={eta}'
            try:
                best = None
                for th in ('4', '8', '16', '32'):
                    k = cg.kappa(mg.round(6).tolist(), [str(round(x, 6)) for x in q0[:-1]] + [str(round(1 - sum(round(x, 6) for x in q0[:-1]), 6))], str(zeta), 1, '2.25', '0.05', eta, th)
                    if best is None or float(k['kappa'].upper()) < best[0]: best = (float(k['kappa'].upper()), th, bool(k['certified']))
                cert[key] = dict(kappa_upper=best[0], theta=best[1], verified=best[2])
            except ValueError as e: cert[key] = str(e)
            print(top, key, cert[key], flush=True)
    out[f'top={top}'] = cert
json.dump(out, open('calib_season1_cert_fresh.json', 'w'), indent=1)
