"""Data-calibrated version of the auction mean-field solver.
Background rival law q0 on a support mgrid that may extend above the top bid; endogenous law on bids {0,h,2h}
(indices 0,1,2 of mgrid); regime-specific causal values Delta[s, m]; quality offsets d[s]; score scale zeta;
prior p0 of the high-value regime; optional fixed KL reference rho (policy ∝ rho * exp(score/eps))."""
import numpy as np
from scipy.special import expit, logsumexp
import grid_certificate as gc
import fast_kernels as fk

class CalibModel(gc.Model):
    def __init__(self, np_grid, nt, eta, eps, T, lam, mgrid, q0, zeta, d, Delta, p0=.5, rho=None):
        super().__init__(np_grid, nt, eta, eps, T, lam)
        self.mgrid = np.asarray(mgrid, float); self.K = len(mgrid); self.base = np.asarray(q0, float)
        self.noise = zeta; self.delta = np.asarray(d, float); self.value = np.asarray(Delta, float)
        self.w = np.zeros((2, self.nact, self.K))
        for s in range(2):
            for ia, a in enumerate(self.actions):
                if ia > 0: self.w[s, ia] = expit((a - self.mgrid - self.delta[s]) / zeta)
        self.p0 = p0; self.logrho = None if rho is None else np.log(np.asarray(rho, float))
    def population_market(self, qt):
        mix = (1 - self.eta) * self.base.copy(); mix[:3] += self.eta * qt
        ws = np.einsum('sam,m->sa', self.w, mix)
        gs = np.einsum('sam,sm,m->sa', self.w, self.value, mix) - self.actions[None, :] * ws
        p = self.p[:, None]
        pw = (1 - p) * ws[0][None, :] + p * ws[1][None, :]
        g = (1 - p) * gs[0][None, :] + p * gs[1][None, :]
        postW = np.clip(p * ws[1][None, :] / np.maximum(pw, 1e-100), 0, 1)
        postL = np.clip(p * (1 - ws[1][None, :]) / np.maximum(1 - pw, 1e-100), 0, 1)
        pw[:, 0] = 0; g[:, 0] = 0; postW[:, 0] = self.p; postL[:, 0] = self.p
        return pw, g, postW, postL
    def backward(self, qflow):
        V = np.zeros((self.nt + 1, self.nb, self.np)); pol = np.zeros((self.nt, self.nb, self.np, self.nact))
        lr = 0. if self.logrho is None else self.logrho
        for t in range(self.nt - 1, -1, -1):
            score = self.scores(V[t + 1], qflow[t])
            adv = score - V[t + 1, :, :, None]
            z = adv / self.entropy + lr
            # infeasible actions carry -1e6 scores -> zero weight
            if fk.HAVE_NUMBA:
                pol[t], lse = fk.gibbs_kernel(np.ascontiguousarray(z)); lse = lse[..., None]
            else:
                lse = logsumexp(z, axis=-1, keepdims=True)
                pol[t] = np.exp(z - lse)
            if self.logrho is not None:   # renormalize reference over feasible actions
                feas = score > -1e5
                lnorm = np.log(np.sum(np.exp(lr) * feas, axis=-1, keepdims=True))
                pol[t] = pol[t]  # Gibbs over feasible set is invariant to the normalizer
                hj = self.entropy * (lse - lnorm)[..., 0]
            else:
                hj = self.entropy * lse[..., 0]
            V[t] = V[t + 1] + self.dt * (self.generator(V[t + 1]) + self.lam * hj)
        return V, pol
    def forward(self, qflow, pol):
        # start all mass at full budget and the prior belief p0 (split between neighbouring nodes)
        j = min(int(np.floor(self.p0 / self.h)), self.np - 2); fr = (self.p0 - self.p[j]) / self.h
        self._init = (j, fr)
        mass, qnew = self._forward(qflow, pol)
        return mass, qnew
    def _forward(self, qflow, pol):
        import types
        m0 = np.zeros((self.nb, self.np)); j, fr = self._init; m0[-1, j] = 1 - fr; m0[-1, j + 1] = fr
        mass = np.zeros((self.nt + 1, self.nb, self.np)); mass[0] = m0; qnew = np.zeros_like(qflow)
        for t in range(self.nt):
            m = mass[t]; nextm = m.copy(); qnew[t] = (m[:, :, None] * pol[t]).sum((0, 1))
            up = self.dt * self.up[:-1][None, :] * m[:, :-1]; dn = self.dt * self.down[1:][None, :] * m[:, 1:]
            nextm[:, :-1] -= up; nextm[:, 1:] += up; nextm[:, 1:] -= dn; nextm[:, :-1] += dn
            pw, _, bw, bl = self.population_market(qflow[t]); jump = self.lam * self.dt; nextm -= jump * m
            if fk.HAVE_NUMBA:
                fk.jump_scatter(nextm, m, np.ascontiguousarray(pol[t]), pw, bw, bl, jump, float(self.h), self.p)
                mass[t + 1] = nextm
                continue
            for ib in range(self.nb):
                for ia in range(min(ib, 2) + 1):
                    amounts = jump * m[ib] * pol[t, ib, :, ia]
                    if ia == 0: nextm[ib] += amounts; continue
                    for prob, bels, newb in ((pw[:, ia], bw[:, ia], ib - ia), (1 - pw[:, ia], bl[:, ia], ib)):
                        src = amounts * prob; jf = np.minimum(np.floor(bels / self.h).astype(int), self.np - 2)
                        frc = np.clip((bels - self.p[jf]) / self.h, 0, 1)
                        np.add.at(nextm[newb], jf, src * (1 - frc)); np.add.at(nextm[newb], jf + 1, src * frc)
            mass[t + 1] = nextm
        return mass, qnew
    def value_at_prior(self, v):
        j = min(int(np.floor(self.p0 / self.h)), self.np - 2); fr = (self.p0 - self.p[j]) / self.h
        return float((1 - fr) * v[0, -1, j] + fr * v[0, -1, j + 1])
    def monetary(self, qflow, policy=None):
        v = np.zeros((self.nt + 1, self.nb, self.np))
        for t in range(self.nt - 1, -1, -1):
            sc = self.scores(v[t + 1], qflow[t])
            pay = np.max(sc, axis=-1) - v[t + 1] if policy is None else np.sum(policy[t] * np.where(sc < -1e5, 0., sc - v[t + 1, :, :, None]), axis=-1)
            v[t] = v[t + 1] + self.dt * (self.generator(v[t + 1]) + self.lam * pay)
        return self.value_at_prior(v)
