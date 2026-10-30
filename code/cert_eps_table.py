"""Arb bounds on kappa_theta across entropy weights eps (Table 3 / Figure 2 left); writes eps_table_arb.json.
Uses the theta recorded for each cell and 4000 time cells, as for Corollary 1."""
import json, sys
import cert_interval as ci
nt = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
saved = json.load(open('eps_table_arb.json'))
out = {}
for key, v in saved.items():
    eta, eps = key.split(','); th = v['theta']
    r = ci.sharp_kappa(1, 2.25, eps, eta, th, nt=nt)
    out[key] = dict(kappa_upper=float(r['kappa'].upper()), theta=th, certified=bool(r['certified']), nt=nt)
    print(key, out[key], flush=True)
json.dump(out, open('eps_table_arb_fresh.json', 'w'), indent=1)
