"""Optional JAX version of the discretized response map Phi_h, for exact Jacobians.

index_test.py and index_norm.py need D Phi_h in tangent coordinates (q1, q2 at each time step,
q0 = 1 - q1 - q2).  The original code uses one-sided finite differences with step 1e-6, which costs
2*nt+1 solves and carries an O(1e-6) truncation error.  Here the same scheme (verify_auction.Auction:
backward Euler on the value, softmax policy, forward mass transport with linear interpolation in the
belief) is written in JAX and differentiated with forward-mode autodiff, in one compiled call.

The map is piecewise smooth: belief interpolation has kinks where a Bayes update crosses a grid node.
Away from kinks autodiff returns the exact derivative of the scheme; at a kink it returns a one-sided
derivative, as finite differences do.  Without JAX, or with JAC=fd, index_test.jac falls back to
finite differences.  Floating-point numerics only; no interval guarantee.
"""
import os

HAVE_JAX = False
if os.environ.get('JAC', 'jax') != 'fd':
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp
        HAVE_JAX = True
    except ImportError:
        pass

_CACHE = {}


def _build(m):
    """Compile Phi_h and its Jacobian for the constants of model m (a grid_certificate.Model)."""
    nb, npg, nact, nt = m.nb, m.np, m.nact, m.nt
    p = jnp.asarray(m.p); h = float(m.h); up = jnp.asarray(m.up); down = jnp.asarray(m.down)
    W = jnp.asarray(m.w); val = jnp.asarray(m.value); base = jnp.asarray(m.base)
    acts = jnp.asarray(m.actions); eta, eps, lam, dt = float(m.eta), float(m.entropy), float(m.lam), float(m.dt)
    jump = lam * dt
    mid = npg // 2

    def market(qt):
        mix = (1 - eta) * base + eta * qt
        ws = jnp.einsum('sac,c->sa', W, mix)
        gs = jnp.einsum('sac,sc,c->sa', W, val, mix) - acts[None, :] * ws
        pc = p[:, None]
        pw = (1 - pc) * ws[0][None, :] + pc * ws[1][None, :]
        g = (1 - pc) * gs[0][None, :] + pc * gs[1][None, :]
        postW = jnp.clip(pc * ws[1][None, :] / jnp.maximum(pw, 1e-100), 0, 1)
        postL = jnp.clip(pc * (1 - ws[1][None, :]) / jnp.maximum(1 - pw, 1e-100), 0, 1)
        pw = pw.at[:, 0].set(0.); g = g.at[:, 0].set(0.)
        postW = postW.at[:, 0].set(p); postL = postL.at[:, 0].set(p)
        return pw, g, postW, postL

    def locate(bel):
        jf = jnp.minimum(jnp.floor(bel / h).astype(jnp.int32), npg - 2)
        fr = jnp.clip((bel - p[jf]) / h, 0, 1)
        return jf, fr

    def interp(row, bel):
        jf, fr = locate(bel)
        return (1 - fr) * row[jf] + fr * row[jf + 1]

    def generator(v):
        res = jnp.zeros_like(v)
        res = res.at[:, :-1].add(up[:-1][None, :] * (v[:, 1:] - v[:, :-1]))
        res = res.at[:, 1:].add(down[1:][None, :] * (v[:, :-1] - v[:, 1:]))
        return res

    def scores(v, qt):
        pw, g, bw, bl = market(qt)
        sc = jnp.full((nb, npg, nact), -1e6)
        for ib in range(nb):
            for ia in range(min(ib, 2) + 1):
                if ia == 0:
                    sc = sc.at[ib, :, 0].set(v[ib]); continue
                vw = interp(v[ib - ia], bw[:, ia]); vl = interp(v[ib], bl[:, ia])
                sc = sc.at[ib, :, ia].set(g[:, ia] + pw[:, ia] * vw + (1 - pw[:, ia]) * vl)
        return sc

    def back_step(V, qt):
        adv = scores(V, qt) - V[:, :, None]
        z = adv / eps
        lse = logsumexp(z, axis=-1, keepdims=True)
        pol = jnp.exp(z - lse)
        Vn = V + dt * (generator(V) + lam * eps * lse[..., 0])
        return Vn, pol

    def fwd_step(mm, xs):
        qt, pol_t = xs
        qn = (mm[:, :, None] * pol_t).sum((0, 1))
        nm = mm
        upm = dt * up[:-1][None, :] * mm[:, :-1]; dnm = dt * down[1:][None, :] * mm[:, 1:]
        nm = nm.at[:, :-1].add(-upm).at[:, 1:].add(upm).at[:, 1:].add(-dnm).at[:, :-1].add(dnm)
        pw, _, bw, bl = market(qt)
        nm = nm - jump * mm
        for ib in range(nb):
            for ia in range(min(ib, 2) + 1):
                amt = jump * mm[ib] * pol_t[ib, :, ia]
                if ia == 0:
                    nm = nm.at[ib].add(amt); continue
                for prob, bel, newb in ((pw[:, ia], bw[:, ia], ib - ia), (1 - pw[:, ia], bl[:, ia], ib)):
                    src = amt * prob; jf, fr = locate(bel)
                    nm = nm.at[newb, jf].add(src * (1 - fr)).at[newb, jf + 1].add(src * fr)
        return nm, qn

    def phi(q):
        V0 = jnp.zeros((nb, npg))
        _, pols = jax.lax.scan(back_step, V0, q, reverse=True)
        m0 = jnp.zeros((nb, npg)).at[-1, mid].set(1.)
        _, qn = jax.lax.scan(fwd_step, m0, (q, pols))
        return qn

    def tangent_phi(x, q):
        # x[2t+a-1] perturbs q[t,a] (a=1,2) with the opposite change in q[t,0], as in index_test.jac
        dx = x.reshape(nt, 2)
        qq = q.at[:, 1:].add(dx).at[:, 0].add(-dx.sum(1))
        return phi(qq)[:, 1:].ravel()

    jac = jax.jit(lambda q: jax.jacfwd(tangent_phi)(jnp.zeros(2 * nt), q))
    return jax.jit(phi), jac


def _key(m):
    import numpy as np
    return (m.nb, m.np, m.nact, m.nt, float(m.eta), float(m.entropy), float(m.lam), float(m.dt),
            np.asarray(m.w).tobytes(), np.asarray(m.value).tobytes(), np.asarray(m.base).tobytes())


def jacobian(m, q):
    """Exact Jacobian of Phi_h at flow q in index_test's tangent coordinates, shape (2nt, 2nt)."""
    import numpy as np
    k = _key(m)
    if k not in _CACHE:
        _CACHE[k] = _build(m)
    return np.asarray(_CACHE[k][1](jnp.asarray(q, dtype=jnp.float64)))


def phi(m, q):
    import numpy as np
    k = _key(m)
    if k not in _CACHE:
        _CACHE[k] = _build(m)
    return np.asarray(_CACHE[k][0](jnp.asarray(q, dtype=jnp.float64)))
