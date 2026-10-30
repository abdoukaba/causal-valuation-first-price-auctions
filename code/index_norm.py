"""(i) Induced L^inf_t(l1) operator norm of DPhi_h on the tangent space (the norm in which kappa_theta is stated),
along the equilibrium path and at random flows; (ii) size distribution of FOSD-isotonicity violations."""
import json, itertools, numpy as np, grid_certificate as gc
from index_test import jac, phi
import monotone_test as mt
EXT = np.array([[1,0],[0,1],[1,-1]])/2.; EXT = np.vstack([EXT,-EXT])          # extreme points of zero-sum l1 unit ball
SIGNS = np.array(list(itertools.product([-1,1],repeat=3)))
def induced_norm(J, nt):
    Jr = J.reshape(nt,2,nt,2)                                   # out t, out comp(1,2), in tau, in comp(1,2)
    best = 0.
    for t in range(nt):
        blk = Jr[t]                                             # (2, nt, 2)
        outs = np.einsum('aTb,eb->Tea', blk, EXT)               # tau, extreme, (y1,y2)
        full = np.stack([-(outs[...,0]+outs[...,1]), outs[...,0], outs[...,1]], -1)   # tau, e, 3
        for s in SIGNS:
            best = max(best, (full @ s).max(axis=1).sum())
    return float(best)
out = {'path': [], 'random': [], 'violations': {}}
for eps in (0.07, 0.05, 0.02, 0.01, 0.005):
    m = gc.Model(21, 60, .5, eps, 1., 2.25); sol = m.solve(0, max_iter=800, damping=.5)
    out['path'].append(dict(eps=eps, norm=induced_norm(jac(m, sol['q']), m.nt))); print(out['path'][-1], flush=True)
for eps in (0.05, 0.01):
    m = gc.Model(21, 60, .5, eps, 1., 2.25); rng = np.random.default_rng(5)
    for k in range(8):
        qr = .98 * rng.dirichlet([rng.choice([.3, 1., 5.])]*3, size=m.nt) + .02/3
        out['random'].append(dict(eps=eps, norm=induced_norm(jac(m, qr), m.nt))); print(out['random'][-1], flush=True)
for eps in (0.05, 0.03, 0.02, 0.01):
    m = gc.Model(21, 60, .5, eps, 1., 2.25); rng = np.random.default_rng(11); mins = []
    for k in range(40):
        q = rng.dirichlet([1,1,1], size=m.nt); qt = mt.shift_down(rng, q, rng.choice([.05,.3,1.]))
        dlt = mt.upper_tails(phi(m, q)) - mt.upper_tails(phi(m, qt)); mins.append(float(dlt.min()))
    mins = np.array(mins)
    out['violations'][str(eps)] = dict(n=40, n_violating=int((mins < -1e-12).sum()), min=float(mins.min()),
                                       q50=float(np.median(mins)), q90_most_negative=float(np.quantile(mins, .1)))
    print(eps, out['violations'][str(eps)], flush=True)
json.dump(out, open('index_norm.json', 'w'), indent=1)
