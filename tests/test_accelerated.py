"""Checks that the optional accelerated paths reproduce the reference numpy code.
Run from code/:  python ../tests/test_accelerated.py
  (1) numba kernels vs the numpy loops in verify_auction.py / calib_model.py (backward, forward, solve);
  (2) the JAX response map and its autodiff Jacobian vs numpy Phi_h and finite differences.
Each part is skipped if its package is missing."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import fast_kernels as fk, jax_phi, grid_certificate as gc, calib_model as cm

rng = np.random.default_rng(7)


def models():
    yield 'Table1 long', gc.Model(21, 60, .5, .05, 1., 2.25)
    yield 'Table1 short', gc.Model(41, 48, .5, .5, .2, .1)
    MG = np.array([0, .05, .10, .15]); D = np.outer([.05, .12], [.6, 1., 1.4, 1.4])
    yield 'calibrated+KL', cm.CalibModel(21, 60, .45, .05, 1., 2.25, MG, [.4, .25, .02, .33], .011,
                                         [.07, -.07], D, .5, np.array([.05, .9, .05]))


def run(m, q, use):
    fk.HAVE_NUMBA = use
    V, pol = m.backward(q); mass, qn = m.forward(q, pol)
    return V, pol, mass, qn


if fk.HAVE_NUMBA:
    for name, m in models():
        q = rng.dirichlet([1, 1, 1], size=m.nt)
        ref = run(m, q, False); t0 = time.time(); ref = run(m, q, False); tn = time.time() - t0
        run(m, q, True); t0 = time.time(); fast = run(m, q, True); tf = time.time() - t0
        err = max(float(np.abs(a - b).max()) for a, b in zip(ref, fast))
        assert err < 1e-12, (name, err)
        print(f'numba  {name:14s} max |diff| = {err:.1e}   one Phi_h: numpy {tn*1e3:.0f} ms, numba {tf*1e3:.1f} ms')
    m = gc.Model(21, 60, .5, .05, 1., 2.25)
    fk.HAVE_NUMBA = False; s0 = m.solve(0, max_iter=800, damping=.5)
    fk.HAVE_NUMBA = True; s1 = m.solve(0, max_iter=800, damping=.5)
    err = float(np.abs(s0['q'] - s1['q']).max()); assert err < 1e-12 and s0['steps'] == s1['steps']
    print(f'numba  Picard solve: same step count ({s1["steps"]}), max |diff q| = {err:.1e}')
else:
    print('numba not installed: skipped')

def fd_jac(m, q, h=1e-6):          # the original finite-difference Jacobian of index_test.py
    phi = lambda qq: m.forward(qq, m.backward(qq)[1])[1]
    base = phi(q)[:, 1:].ravel(); J = np.zeros((2 * m.nt, 2 * m.nt))
    for t in range(m.nt):
        for a in (1, 2):
            qq = q.copy(); qq[t, a] += h; qq[t, 0] -= h
            J[:, 2 * t + a - 1] = (phi(qq)[:, 1:].ravel() - base) / h
    return J


if jax_phi.HAVE_JAX:
    for eps in (0.05, 0.01):
        m = gc.Model(21, 60, .5, eps, 1., 2.25)
        q = .98 * rng.dirichlet([1, 1, 1], size=m.nt) + .02 / 3
        e_phi = float(np.abs(jax_phi.phi(m, q) - m.forward(q, m.backward(q)[1])[1]).max())
        assert e_phi < 1e-12, e_phi
        jax_phi.jacobian(m, q); t0 = time.time(); Ja = jax_phi.jacobian(m, q); ta = time.time() - t0
        t0 = time.time(); Jf = fd_jac(m, q); tf = time.time() - t0
        rel = float(np.abs(Ja - Jf).max() / np.abs(Jf).max())
        rho = lambda J: float(np.abs(np.linalg.eigvals(J)).max())
        assert rel < 1e-3, rel
        print(f'JAX    eps={eps}: |Phi_jax-Phi_np| = {e_phi:.1e}; Jacobian rel. diff vs FD = {rel:.1e}; '
              f'spectral radius {rho(Ja):.6f} (FD {rho(Jf):.6f}); time {ta:.2f} s vs FD {tf:.1f} s')
else:
    print('JAX not installed: skipped')
