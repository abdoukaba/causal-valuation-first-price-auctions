"""Index/degree route to uniqueness (Govindan-Wilson): Phi_h maps the product of simplices into its interior
(Gibbs policies have full support), so deg(id-Phi_h)=1 and #fixed points = 1 whenever det(I-DPhi_h)>0 at every
fixed point; a sufficient global condition is that DPhi_h(q) has no real eigenvalue >= 1 for any q.
We compute DPhi_h by finite differences in tangent coordinates (q1,q2 per time step) and report
(i) along the equilibrium continuation path in eps, (ii) at random flows.  Non-certified numerics."""
import json, sys, time, numpy as np
import grid_certificate as gc
def phi(m, q):
    return m.forward(q, m.backward(q)[1])[1]
def jac(m, q, h=1e-6):
    """D Phi_h in tangent coordinates: exact (JAX autodiff, jax_phi.py) if JAX is installed and JAC!=fd,
    otherwise one-sided finite differences with step h (the original method)."""
    import jax_phi
    if jax_phi.HAVE_JAX:
        return jax_phi.jacobian(m, q)
    nt = m.nt; base = phi(m, q)[:, 1:].ravel(); J = np.zeros((2 * nt, 2 * nt))
    for t in range(nt):
        for a in (1, 2):
            qq = q.copy(); qq[t, a] += h; qq[t, 0] -= h
            J[:, 2 * t + a - 1] = (phi(m, qq)[:, 1:].ravel() - base) / h
    return J
def stats(J):
    ev = np.linalg.eigvals(J); real = ev[np.abs(ev.imag) < 1e-9].real
    sign, logdet = np.linalg.slogdet(np.eye(len(J)) - J)
    return dict(spectral_radius=float(np.abs(ev).max()), max_real_eig=float(real.max()) if len(real) else None,
                max_re=float(ev.real.max()), det_sign=int(sign), opnorm_inf=float(np.abs(J).sum(1).max()))
import os
out = {'path': [], 'random': {}}
q = None
for eps in (() if os.environ.get('RANDOM_ONLY') else (0.07, 0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005)):
    m = gc.Model(21, 60, .5, eps, 1., 2.25)
    sol = m.solve(0, max_iter=800, damping=.5)
    t0 = time.time(); s = stats(jac(m, sol['q'])); s.update(eps=eps, residual=sol['res'], secs=time.time() - t0)
    out['path'].append(s); print('path', s, flush=True)
for eps in (0.05, 0.01):
    m = gc.Model(21, 60, .5, eps, 1., 2.25); rng = np.random.default_rng(3); rs = []
    for k in range(int(sys.argv[1]) if len(sys.argv) > 1 else 12):
        alpha = rng.choice([.3, 1., 5.]); qr = rng.dirichlet([alpha] * 3, size=m.nt)
        qr = .98 * qr + .02 / 3                       # keep finite differences inside the simplex
        rs.append(stats(jac(m, qr)))
    out['random'][str(eps)] = dict(n=len(rs), max_real_eig=max((r['max_real_eig'] for r in rs if r['max_real_eig'] is not None), default=None), max_re=max(r['max_re'] for r in rs),
                                   max_spectral_radius=max(r['spectral_radius'] for r in rs), all_det_positive=all(r['det_sign'] > 0 for r in rs),
                                   max_opnorm=max(r['opnorm_inf'] for r in rs))
    print('random', eps, out['random'][str(eps)], flush=True)
json.dump(out, open(os.environ.get('OUT','index_test.json'), 'w'), indent=1)
