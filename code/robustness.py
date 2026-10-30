"""Robustness of uniqueness evidence and certificate to the mechanism (score noise zeta -> hard
first-price rule), the background rival law q0, and a reserve price.  Original horizon
(T=1, lambda=2.25, eps=0.05).  Certificate column: Arb interval bound of Theorem 2 (best theta
over a small grid) when its positivity hypothesis w(1-w)>=1/8 is provable; numerics: 21 belief
nodes x 60 steps, 6 random Picard starts, non-certified."""
import json, sys, numpy as np
from scipy.special import expit
import grid_certificate as gc, cert_interval as ci
from flint import arb


def model(eta, zeta, q0, reserve=0.0):
    m = gc.Model(21, 60, eta, .05, 1., 2.25)
    m.noise = zeta; m.base = np.array(q0)
    for s in range(2):
        for ia, a in enumerate(m.actions):
            if ia > 0:
                m.w[s, ia] = expit((a - m.actions - m.delta[s]) / zeta) * (a >= reserve - 1e-12)
    return m


def cert(eta, zeta, q0):
    ci.Q0 = [arb(str(x)) for x in q0]
    old = ci.primitives
    def prim(e, zeta_=None): return old(e, zeta=arb(str(zeta)))
    ci.primitives = prim
    best = None
    try:
        for th in ('4', '8', '12', '16', '20', '30'):
            r = ci.sharp_kappa(1, 2.25, '0.05', str(eta), th, nt=2000)
            if best is None or r['kappa'].mid() < best['kappa'].mid(): best = r
        res = dict(kappa=best['kappa'].str(6), certified=best['certified'])
    except ValueError as e:
        res = dict(kappa=None, certified=False, note='positivity hypothesis fails: ' + str(e))
    finally:
        ci.primitives = old; ci.Q0 = [arb('0.2'), arb('0.4'), arb('0.4')]
    return res


rows = []
cases = [('baseline', .45, .2, (.2, .4, .4), 0), ('baseline', .5, .2, (.2, .4, .4), 0),
         ('zeta=0.1', .45, .1, (.2, .4, .4), 0), ('zeta=0.05', .45, .05, (.2, .4, .4), 0),
         ('zeta=0.02 (near hard rule)', .45, .02, (.2, .4, .4), 0),
         ('q0 low-bid', .45, .2, (.6, .3, .1), 0), ('q0 high-bid', .45, .2, (.1, .3, .6), 0),
         ('reserve 0.075', .45, .2, (.2, .4, .4), .075)]
for name, eta, zeta, q0, res in cases:
    m = model(eta, zeta, q0, res)
    sols = [m.solve(i, max_iter=800, damping=.5) for i in range(6)]
    spread = max(float(abs(sols[0]['q'] - s['q']).max()) for s in sols)
    mon = m.monetary_policy_value(sols[0]['q'], sols[0]['pol']); br = m.monetary_policy_value(sols[0]['q'], None)
    c = cert(eta, zeta, q0) if res == 0 else dict(kappa=None, certified=False, note='reserve violates positivity (w=0 for a<reserve)')
    row = dict(case=name, eta=eta, zeta=zeta, q0=q0, reserve=res, residual=max(s['res'] for s in sols), steps=max(s['steps'] for s in sols),
               multistart_spread=spread, mon=mon, br=br, br_ratio=br / mon if mon > 0 else None, **{'cert_' + k: v for k, v in c.items()})
    rows.append(row); print(row, flush=True)
json.dump(rows, open('robustness.json', 'w'), indent=1, default=float)
