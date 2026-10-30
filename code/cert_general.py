"""cert_interval.sharp_kappa with a configurable background rival law q0 on a support that may extend above the
top bid, and a configurable score scale zeta.  The endogenous law still lives on the bid grid {0,h,2h}."""
from flint import arb
import cert_interval as ci
A, amax, amin = ci.A, ci.amax, ci.amin
def make_primitives(mgrid, q0, bid_idx=(0, 1, 2)):
    MG = [A(x) for x in mgrid]; Q0 = [A(x) for x in q0]; K = len(MG)
    def primitives(eta, zeta=A('0.2')):
        w = {(s, a, m): 1 / (1 + (-(ci.ACTS[a] - MG[m] - ci.D_OFF[s]) / zeta).exp()) for s in range(2) for a in range(2) for m in range(K)}
        g = {(s, a, m): w[s, a, m] * (ci.BASE[s] + ci.SLOPE * MG[m] - ci.ACTS[a]) for s in range(2) for a in range(2) for m in range(K)}
        verts = [[(1 - eta) * Q0[j] + (eta if j == bid_idx[k] else 0) for j in range(K)] for k in range(3)]
        pc = {}; LPsi = A(1); rplus = A(0)
        for a in range(2):
            ell = [[sum((v[m] * w[s, a, m] for m in range(K)), A(0)) for v in verts] for s in range(2)]
            lmin = amin([x for r in ell for x in r]); lmax = amax([x for r in ell for x in r])
            lo = [amin(ell[s]) for s in range(2)]; hi = [amax(ell[s]) for s in range(2)]
            gamma = amax([amax([g[s, a, m] for m in range(K)]) - amin([g[s, a, m] for m in range(K)]) for s in range(2)])
            omega = amax([amax([w[s, a, m] for m in range(K)]) - amin([w[s, a, m] for m in range(K)]) for s in range(2)])
            O1s, O0s = [], []
            for i in range(K):
                for j in range(K):
                    d1 = w[1, a, i] - w[1, a, j]; d0 = w[0, a, i] - w[0, a, j]
                    for l0 in (lo[0], hi[0]):
                        for l1 in (lo[1], hi[1]):
                            O1s.append(abs(l0 * d1 - l1 * d0)); O0s.append(abs((1 - l0) * d1 - (1 - l1) * d0))
            beta = amax(O1s) / (4 * lmin) + amax(O0s) / (4 * (1 - lmax))
            Lg = amax([abs(g[1, a, m] - g[0, a, m]) for m in range(K)]); Lw = amax([abs(w[1, a, m] - w[0, a, m]) for m in range(K)])
            for m in range(K):
                LPsi = amax([LPsi, w[1, a, m] / w[0, a, m], w[0, a, m] / w[1, a, m], (1 - w[1, a, m]) / (1 - w[0, a, m]), (1 - w[0, a, m]) / (1 - w[1, a, m])])
            rplus = amax([rplus] + [g[s, a, m] for s in range(2) for m in range(K)])
            if not (bool(lmin > A('0.1465')) and bool(lmax < A('0.8535'))):
                raise ValueError(f'Assumption 2 fails: lmin={float(lmin.mid()):.4f}, lmax={float(lmax.mid()):.4f}')
            pc[a] = dict(gamma=gamma, omega=omega, beta=beta, Lg=Lg, Lw=Lw, lmin=lmin, lmax=lmax)
        return pc, LPsi, rplus, None
    return primitives
def kappa(mgrid, q0, zeta, T, lam, eps, eta, theta, nt=2000):
    prim = make_primitives(mgrid, q0); z = A(zeta)
    old = ci.primitives; ci.primitives = lambda e: prim(e, z)
    try: return ci.sharp_kappa(T, lam, eps, eta, theta, nt=nt)
    finally: ci.primitives = old
