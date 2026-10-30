"""Interval-arithmetic (Arb ball arithmetic, python-flint) enclosure of the contraction bounds.

Every real number is an Arb ball that rigorously contains the exact value; every operation is
outward-rounded by Arb.  The time integration uses cell-wise upper bounds that are rigorous
because V̄, L, C_V are nonincreasing in t (they solve backward ODEs with nonnegative sources)
and D_a, P, Λ are nondecreasing functions of them; M_θ is nondecreasing in t.  On each cell
[t_k, t_{k+1}] the supremum of M_θ(t)max(1,Λ(t)/θ)+P(t) is bounded by M_θ(t_{k+1})max(1,Λ(t_k)/θ)+P(t_k).
A bound is CERTIFIED iff Arb proves (ball < 1).
"""
from __future__ import annotations
import json, sys
from flint import arb, ctx
ctx.prec = 80

def A(x): return arb(str(x))          # decimal string -> enclosing ball
def amax(xs):
    m = xs[0]
    for x in xs[1:]: m = m.max(x)
    return m
def amin(xs):
    m = xs[0]
    for x in xs[1:]: m = m.min(x)
    return m

ACTS = [A('0.05'), A('0.10')]              # positive bids a=h,2h
MGRID = [A('0'), A('0.05'), A('0.10')]
Q0 = [A('0.2'), A('0.4'), A('0.4')]
D_OFF = [A('-0.01'), A('0.01')]
BASE = [A('0.060'), A('0.095')]
SLOPE = A('0.03')


def primitives(eta, zeta=A('0.2')):
    w = {(s, a, m): 1 / (1 + (-(ACTS[a] - MGRID[m] - D_OFF[s]) / zeta).exp())
         for s in range(2) for a in range(2) for m in range(3)}
    g = {(s, a, m): w[s, a, m] * (BASE[s] + SLOPE * MGRID[m] - ACTS[a]) for s in range(2) for a in range(2) for m in range(3)}
    verts = [[(1 - eta) * Q0[j] + (eta if j == k else 0) for j in range(3)] for k in range(3)]
    pc = {}
    LPsi = A(1); rplus = A(0)
    for a in range(2):
        ell = [[sum((v[m] * w[s, a, m] for m in range(3)), A(0)) for v in verts] for s in range(2)]
        lmin = amin([x for r in ell for x in r]); lmax = amax([x for r in ell for x in r])
        lo = [amin(ell[s]) for s in range(2)]; hi = [amax(ell[s]) for s in range(2)]
        gamma = amax([amax([g[s, a, m] for m in range(3)]) - amin([g[s, a, m] for m in range(3)]) for s in range(2)])
        omega = amax([amax([w[s, a, m] for m in range(3)]) - amin([w[s, a, m] for m in range(3)]) for s in range(2)])
        O1s, O0s = [], []
        for i in range(3):
            for j in range(3):
                d1 = w[1, a, i] - w[1, a, j]; d0 = w[0, a, i] - w[0, a, j]
                for l0 in (lo[0], hi[0]):
                    for l1 in (lo[1], hi[1]):
                        O1s.append(abs(l0 * d1 - l1 * d0)); O0s.append(abs((1 - l0) * d1 - (1 - l1) * d0))
        O1, O0 = amax(O1s), amax(O0s)
        beta = O1 / (4 * lmin) + O0 / (4 * (1 - lmax))
        Lg = amax([abs(g[1, a, m] - g[0, a, m]) for m in range(3)])
        Lw = amax([abs(w[1, a, m] - w[0, a, m]) for m in range(3)])
        for m in range(3):
            LPsi = amax([LPsi, w[1, a, m] / w[0, a, m], w[0, a, m] / w[1, a, m],
                         (1 - w[1, a, m]) / (1 - w[0, a, m]), (1 - w[0, a, m]) / (1 - w[1, a, m])])
        rplus = amax([rplus] + [g[s, a, m] for s in range(2) for m in range(3)])
        if not (bool(lmin > A('0.1465')) and bool(lmax < A('0.8535'))):
            raise ValueError('w(1-w)>=1/8 not proven')
        pc[a] = dict(gamma=gamma, omega=omega, beta=beta, Lg=Lg, Lw=Lw, lmin=lmin, lmax=lmax)
    wmin = amin([amin([w[k], 1 - w[k]]) for k in w])
    return pc, LPsi, rplus, wmin


def sharp_kappa(T, lam, eps, eta, theta, nt=4000, nb=5):
    T, lam, eps, eta, theta = A(T), A(lam), A(eps), A(eta), A(theta)
    pc, LPsi, rplus, _ = primitives(eta)
    dt = T / nt; logJ = arb(3).log()
    Lw2 = amax([c['Lw'] ** 2 for c in pc.values()]); Lwmax = amax([c['Lw'] for c in pc.values()])
    g = lam * (LPsi - 1 + 2 * Lw2)
    Vbar = [lam * (T - k * dt) * (rplus + eps * logJ) for k in range(nt + 1)]
    L = [A(0)] * (nt + 1)
    CV = [[A(0)] * nb for _ in range(nt + 1)]
    egdt = (g * dt).exp()
    for k in range(nt - 1, -1, -1):
        S0 = amax([c['Lg'] + c['Lw'] * Vbar[k] for c in pc.values()])
        L[k] = L[k + 1] * egdt + lam * S0 * (egdt - 1) / g
        Da = {a: eta / 2 * (c['gamma'] + c['omega'] * (Vbar[k] + 2 * L[k] * c['Lw']) + L[k] * c['beta']) for a, c in pc.items()}
        for b in range(nb):
            feas = [a for a in (0, 1) if a + 1 <= b]
            CV[k][b] = CV[k + 1][b] + (lam * dt * amax([Da[a] for a in feas]) if feas else A(0))
    P = []; Lam = []
    for k in range(nt + 1):
        Dk = {a: eta / 2 * (c['gamma'] + c['omega'] * (Vbar[k] + 2 * L[k] * c['Lw']) + L[k] * c['beta']) for a, c in pc.items()}
        terms = [Dk[a] + pc[a]['lmin'] * CV[k][b - (a + 1)] + (2 - pc[a]['lmin']) * CV[k][b]
                 for b in range(nb) for a in (0, 1) if a + 1 <= b]
        P.append(amax(terms) / eps)
        Lam.append(amax([c['Lg'] + c['Lw'] * (Vbar[k] + 2 * L[k] * c['Lw']) + (LPsi + 1) * L[k] for c in pc.values()]) / eps)
    K = eta * amax([c['omega'] + theta * c['beta'] / 2 for c in pc.values()])
    M = [A(0)] * (nt + 1)
    for k in range(nt):
        ck = lam * ((LPsi - 1) + (2 * Lwmax + Lam[k]) / theta)
        e = (ck * dt).exp()
        M[k + 1] = M[k] * e + lam * (P[k] + K) * (e - 1) / ck
    cand = [M[k + 1] * amax([A(1), Lam[k] / theta]) + P[k] for k in range(nt)]
    cand.append(M[nt] * amax([A(1), Lam[nt] / theta]) + P[nt])
    kappa = amax(cand)
    return dict(kappa=kappa, certified=bool(kappa < 1), P0=P[0], Lam0=Lam[0], L0=L[0], CV0=CV[0][nb - 1], K=K, MT=M[nt], LPsi=LPsi, rplus=rplus)


def analytic_kappa_v1(eta='0.5', eps='0.5', T='0.2', lam='0.1', sigma='0.2', c='0.36'):
    eta, eps, T, lam, sigma, c = map(A, (eta, eps, T, lam, sigma, c))
    _, _, _, wmin = primitives(eta)
    if not bool(wmin > c):
        raise ValueError('positivity c not proven')
    R = A('0.2'); J = arb(3)
    M = lam * T * (R + eps * J.log()); Lp = 2 * M; CV = (R + M) * lam * T * eta
    CQ = eta * (R + M + Lp / c) + CV; Cpiq = 2 * CQ / eps
    CQp = 2 * R + 2 * M + Lp / c ** 2; Cpix = amax([A(2), 2 * CQp / eps])
    CKx = 2 + 1 / c ** 2; CKq = eta * (1 + 1 / c)
    alpha = sigma ** 2 / 2 + lam * (CKx - 1 + Cpix)
    Cm = lam * T * (alpha * T).exp() * (CKq + Cpiq)
    kappa = Cpiq + Cpix * Cm
    return dict(kappa=kappa, certified=bool(kappa < 1), wmin=wmin)


def s(x):  # printable upper bound string
    return x.str(10, radius=True)

if __name__ == '__main__':
    out = {}
    for tag, args in [('v1_analytic_eta0.5_eps0.5', dict(eta='0.5', eps='0.5')),
                      ('v1_analytic_eta0.1_eps0.1', dict(eta='0.1', eps='0.1'))]:
        r = analytic_kappa_v1(**args); out[tag] = {k: (s(v) if isinstance(v, arb) else v) for k, v in r.items()}
        print(tag, out[tag])
    cases = [('orig_eta0.45', (1, 2.25, '0.05', '0.45', '16.18')), ('orig_eta0.48', (1, 2.25, '0.05', '0.48', '15')),
             ('orig_eta0.47', (1, 2.25, '0.05', '0.47', '15.5')), ('orig_eta0.485', (1, 2.25, '0.05', '0.485', '16')), ('orig_eta0.487', (1, 2.25, '0.05', '0.487', '16')), ('orig_eta0.40', (1, 2.25, '0.05', '0.40', '17.7')),
             ('orig_eta0.30', (1, 2.25, '0.05', '0.30', '19.36')), ('orig_eta0.50', (1, 2.25, '0.05', '0.50', '16.18')),
             ('short_spec', ('0.2', '0.1', '0.5', '0.5', '1')), ('orig_eta0.5_eps0.07', (1, 2.25, '0.07', '0.5', '16')),
             ('orig_eta0.45_eps0.04', (1, 2.25, '0.04', '0.45', '16'))]
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    for tag, (T, lam, eps, eta, th) in cases:
        r = sharp_kappa(T, lam, eps, eta, th, nt=nt)
        out[tag] = {k: (s(v) if isinstance(v, arb) else v) for k, v in r.items()}
        out[tag]['theta'] = th; out[tag]['nt'] = nt
        print(tag, 'kappa', out[tag]['kappa'], 'certified', r['certified'])
    json.dump(out, open('cert_interval.json', 'w'), indent=1)
