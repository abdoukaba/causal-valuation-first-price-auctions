"""Arb bounds on kappa_theta over a wider grid of entropy weights (Section 5: the criterion is verified at every
tested eps in {0.05,...,1} at eta=0.45 and at every tested eps >= 0.06 at eta=0.5). Best theta over a small grid,
2000 cells. Writes eps_grid_arb.json."""
import json, cert_interval as ci
out = {}
for eta in ('0.45', '0.5'):
    for eps in ('0.05', '0.06', '0.07', '0.08', '0.1', '0.15', '0.2', '0.3', '0.5', '1'):
        best = None
        for th in ('4', '8', '10', '12', '14', '16', '20'):
            r = ci.sharp_kappa(1, 2.25, eps, eta, th, nt=2000); k = float(r['kappa'].upper())
            if best is None or k < best['kappa_upper']: best = dict(kappa_upper=k, theta=th, certified=bool(r['certified']))
        out[f'{eta},{eps}'] = best; print(eta, eps, best, flush=True)
json.dump(out, open('eps_grid_arb.json', 'w'), indent=1)
