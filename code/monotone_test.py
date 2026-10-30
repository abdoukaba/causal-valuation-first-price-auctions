"""Is the mean-field response map Phi monotone (isotone or antitone) in first-order stochastic
dominance (FOSD) on bid flows?  If so, Tarski/Milgrom-Roberts bracketing applies: iterate Phi from the
top flow (all mass on the highest bid) and the bottom flow (all mass on 0); if both converge to the same
flow, the equilibrium is unique (isotone case), with Phi^2 used in the antitone case.
Non-certified numerics on the 21-node grid."""
import json, sys, numpy as np
import grid_certificate as gc

def upper_tails(q):            # FOSD on {0,h,2h}: compare P(bid>=h), P(bid>=2h)
    return np.stack([q[..., 1] + q[..., 2], q[..., 2]], -1)

def compare(q1, q2, tol=1e-12):
    d = upper_tails(q1) - upper_tails(q2)
    if np.all(d >= -tol): return 1
    if np.all(d <= tol): return -1
    return 0

def phi(m, q):
    V, pol = m.backward(q); return m.forward(q, pol)[1]

def shift_down(rng, q, size):
    """Return q~ <= q in FOSD: move a random fraction of mass one bid level down."""
    qt = q.copy()
    f2 = rng.uniform(0, size, len(q)) * qt[:, 2]; qt[:, 2] -= f2; qt[:, 1] += f2
    f1 = rng.uniform(0, size, len(q)) * qt[:, 1]; qt[:, 1] -= f1; qt[:, 0] += f1
    return qt

if __name__ == "__main__":
    out = {}
    for eps in (0.05, 0.02, 0.01):
        for eta in (0.5, 0.9):
            m = gc.Model(21, 60, eta, eps, 1., 2.25)
            rng = np.random.default_rng(0); res = []
            for k in range(40):
                q = rng.dirichlet([1, 1, 1], size=m.nt); qt = shift_down(rng, q, rng.choice([.05, .3, 1.]))
                res.append(compare(phi(m, q), phi(m, qt)))
            res = np.array(res)
            top = np.tile([0, 0, 1.], (m.nt, 1)); bot = np.tile([1., 0, 0], (m.nt, 1))
            it_top, it_bot = top.copy(), bot.copy(); traj = []
            for n in range(60):
                it_top, it_bot = phi(m, it_top), phi(m, it_bot)
                traj.append(float(np.abs(it_top - it_bot).max()))
                if traj[-1] < 1e-12: break
            out[f'eps={eps},eta={eta}'] = dict(isotone=int((res == 1).sum()), antitone=int((res == -1).sum()), neither=int((res == 0).sum()),
                                              extremal_gap_final=traj[-1], iterations=len(traj), gap_first5=traj[:5])
            print(eps, eta, out[f'eps={eps},eta={eta}'], flush=True)
    json.dump(out, open('monotone_test.json', 'w'), indent=1)
