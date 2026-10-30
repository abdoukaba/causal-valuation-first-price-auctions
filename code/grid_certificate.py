"""Grid-exact global contraction bound for the discretized auction MFG.

The bound holds for EVERY pair of population flows q, q~ on the given grid
(it is not a local Jacobian estimate).  It uses only:
  * the primitives (win probabilities, rewards) evaluated on finite sets,
  * monotonicity / sup-nonexpansiveness of the explicit backward scheme,
  * l1-nonexpansiveness of the transposed forward scheme,
  * the softmax bound ||pi(u)-pi(u')||_1 <= osc(u-u')/(2 eps).
Floating-point rounding is covered by a multiplicative safety margin.
Derivation: Section 7 of auction_mfg_v2.pdf (grid_kappa: discrete analogue described there).
"""
from __future__ import annotations
import argparse, json, math, sys
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
import verify_auction as va


class Model(va.Auction):
    def __init__(self, np_grid, nt, eta, eps, T, lam, check=True):
        self._T, self._lam = T, lam
        super().__init__(np_grid, nt, eta, eps)
        self.T, self.lam, self.dt = T, lam, T / nt
        if check:
            assert self.dt * (self.lam + max(self.up + self.down)) < .99

    def certificate(self):  # disable parent's analytic certificate here
        return {}


def primitive_constants(m: Model):
    eta, J = m.eta, 3
    acts = [1, 2]
    W, G = m.w, m.w * (m.value[:, None, :] - m.actions[None, :, None])  # g_s(a,m)
    # range of ell_s(a)=sum_m Q(m) w_s(a,m) over Q=(1-eta)q0+eta*q, q in simplex
    verts = [(1 - eta) * m.base + eta * np.eye(3)[k] for k in range(3)]
    out = {}
    LPsi = 1.0
    for a in acts:
        ell = np.array([[v @ W[s, a] for v in verts] for s in range(2)])  # (s,vertex)
        lmin, lmax = ell.min(), ell.max()
        lo, hi = ell.min(1), ell.max(1)  # box for (ell0, ell1)
        gamma = max(np.ptp(G[s, a]) for s in range(2))
        omega = max(np.ptp(W[s, a]) for s in range(2))
        # O1 = max over box and pairs (m,m') of |ell0 (w1m-w1m') - ell1 (w0m-w0m')|
        O1 = O0 = 0.
        for i in range(3):
            for j in range(3):
                d1, d0 = W[1, a, i] - W[1, a, j], W[0, a, i] - W[0, a, j]
                for l0 in (lo[0], hi[0]):
                    for l1 in (lo[1], hi[1]):
                        O1 = max(O1, abs(l0 * d1 - l1 * d0))
                        O0 = max(O0, abs((1 - l0) * (-d1) - (1 - l1) * (-d0)))
        beta = O1 / (4 * lmin) + O0 / (4 * (1 - lmax))
        g1 = np.abs(G[1, a] - G[0, a]).max()
        w1 = np.abs(W[1, a] - W[0, a]).max()
        ratios = np.concatenate([W[1, a] / W[0, a], W[0, a] / W[1, a],
                                 (1 - W[1, a]) / (1 - W[0, a]), (1 - W[0, a]) / (1 - W[1, a])])
        LPsi = max(LPsi, ratios.max())
        out[a] = dict(gamma=gamma, omega=omega, beta=beta, Lg=g1, Lw=w1, lmin=lmin, lmax=lmax)
    rmax = max(0., float(G[:, 1:, :].max()))
    return out, LPsi, rmax


def grid_kappa(m: Model, margin=1.01):
    pc, LPsi, rmax = primitive_constants(m)
    nt, dt, lam, eps, eta, h = m.nt, m.dt, m.lam, m.entropy, m.eta, m.h
    J = 3
    Vbar = np.zeros(nt + 1); L = np.zeros(nt + 1); CV = np.zeros(nt + 1)
    D = np.zeros(nt)
    for t in range(nt - 1, -1, -1):
        Da = {a: eta / 2 * (c['gamma'] + c['omega'] * (Vbar[t + 1] + 2 * L[t + 1] * c['Lw']) + L[t + 1] * c['beta']) for a, c in pc.items()}
        D[t] = max(Da.values())
        CV[t] = CV[t + 1] + dt * lam * D[t]
        Vbar[t] = Vbar[t + 1] + dt * lam * (rmax + eps * math.log(J))
        lipscore = max(c['Lg'] + c['Lw'] * (Vbar[t + 1] + 2 * L[t + 1] * c['Lw']) + LPsi * L[t + 1] for c in pc.values())
        L[t] = (1 - dt * lam) * L[t + 1] + dt * lam * max(lipscore, L[t + 1])
    P = np.array([(D[t] + 2 * CV[t + 1]) / eps for t in range(nt)])
    Kk = eta * max(c['omega'] + c['beta'] / h for c in pc.values())
    Mm = np.zeros(nt + 1)
    for t in range(nt):
        Mm[t + 1] = Mm[t] + dt * lam * (P[t] + Kk)
    kappa = float(np.max(Mm[:nt] + P)) * margin
    return dict(kappa=kappa, P_max=float(P.max()), mass_term=float(Mm[nt - 1]), K_kernel=Kk,
                CV0=float(CV[0]), Vbar0=float(Vbar[0]), L0=float(L[0]), LPsi=float(LPsi),
                h=h, nt=nt, prim={str(a): c for a, c in pc.items()})


def empirical_ratio(m: Model, pairs=6, seed=0):
    rng = np.random.default_rng(seed); worst = 0.
    for _ in range(pairs):
        q1 = rng.dirichlet([1, 1, 1], size=m.nt); q2 = rng.dirichlet([1, 1, 1], size=m.nt)
        r1 = m.forward(q1, m.backward(q1)[1])[1]; r2 = m.forward(q2, m.backward(q2)[1])[1]
        worst = max(worst, np.abs(r1 - r2).sum(1).max() / np.abs(q1 - q2).sum(1).max())
    return float(worst)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--np', type=int, default=21); ap.add_argument('--nt', type=int, default=60)
    ap.add_argument('--eta', type=float, default=.5); ap.add_argument('--eps', type=float, default=.05)
    ap.add_argument('--T', type=float, default=1.); ap.add_argument('--lam', type=float, default=2.25)
    a = ap.parse_args()
    m = Model(a.np, a.nt, a.eta, a.eps, a.T, a.lam)
    res = grid_kappa(m); res['empirical_ratio'] = empirical_ratio(m)
    print(json.dumps(res, indent=1, default=float))


def continuum_kappa(m: Model, theta: float, margin=1.01):
    """Grid-free (PIDE-level) bound.  Backward quantities from the exact linear ODE
    recursions, evaluated with left-endpoint (larger) values on each sub-interval;
    forward bounded-Lipschitz growth integrated with exp().  Belief grid unused."""
    pc, LPsi, rmax = primitive_constants(m)
    nt, dt, lam, eps, eta, nb = m.nt, m.dt, m.lam, m.entropy, m.eta, m.nb
    J = 3
    Lw = max(c['Lw'] for c in pc.values())
    assert all(c['lmin'] > .1465 and c['lmax'] < .8535 for c in pc.values())  # pw(1-pw)>=1/8
    # exact continuum solutions where available
    tgrid = np.linspace(0, m.T, nt + 1)
    Vbar = lam * (m.T - tgrid) * (rmax + eps * math.log(J))
    L = np.zeros(nt + 1); CV = np.zeros((nt + 1, nb)); P = np.zeros(nt + 1); LipPi = np.zeros(nt + 1)
    for t in range(nt - 1, -1, -1):
        # interval [t, t+1]: use values at node t for Vbar (larger), L from upper Euler step
        Lt = L[t + 1]
        g = lam * (LPsi - 1)
        for _ in range(6):  # fixed-point iteration from below; then outward step below
            lipscore = max(c['Lg'] + c['Lw'] * (Vbar[t] + 2 * Lt * c['Lw']) for c in pc.values())
            Lt = L[t + 1] * math.exp(g * dt) + lam * lipscore * dt * math.exp(g * dt)
        L[t] = Lt * (1 + 1e-9)
        Da = {a: eta / 2 * (c['gamma'] + c['omega'] * (Vbar[t] + 2 * L[t] * c['Lw']) + L[t] * c['beta'])
              for a, c in pc.items()}
        for b in range(nb):
            feas = [a for a in (1, 2) if a <= b]
            CV[t, b] = CV[t + 1, :b + 1].max() + dt * lam * (max(Da[a] for a in feas) if feas else 0.)
    for t in range(nt + 1):
        pt = lp = 0.
        for b in range(nb):
            for a in (1, 2):
                if a > b: continue
                c = pc[a]
                Dat = eta / 2 * (c['gamma'] + c['omega'] * (Vbar[t] + 2 * L[t] * c['Lw']) + L[t] * c['beta'])
                pt = max(pt, Dat + c['lmin'] * CV[t, b - a] + (2 - c['lmin']) * CV[t, b])  # |K~dV - dV| bound, no Lipschitz of dV used
                lp = max(lp, c['Lg'] + c['Lw'] * (Vbar[t] + 2 * L[t] * c['Lw']) + LPsi * L[t] + L[t])
        P[t] = pt / eps; LipPi[t] = lp / eps
    assert np.all(np.diff(P) <= 1e-15) and np.all(np.diff(LipPi) <= 1e-15)  # nonincreasing in t
    Kk = eta * max(c['omega'] + theta * c['beta'] / 2 for c in pc.values())
    Mm = np.zeros(nt + 1)
    for t in range(nt):
        ct = lam * ((LPsi - 1) + (2 * Lw + LipPi[t]) / theta)
        Mm[t + 1] = Mm[t] * math.exp(ct * dt) + dt * lam * (P[t] + Kk) * math.exp(ct * dt)
    fac = np.maximum(1., LipPi / theta)
    kappa = float(np.max(Mm * fac + P)) * margin
    return dict(kappa=kappa, theta=theta, P0=float(P[0]), LipPi0=float(LipPi[0]), K=float(Kk),
                M_T=float(Mm[-1]), CV0=float(CV[0].max()), L0=float(L[0]), Vbar0=float(Vbar[0]), LPsi=float(LPsi))


def best_continuum(m, thetas=np.geomspace(1, 200, 60)):
    rs = [continuum_kappa(m, th) for th in thetas]
    return min(rs, key=lambda r: r['kappa'])
