"""Optional numba kernels for the finite-difference equilibrium solver.

The three inner loops of the solver are compiled with numba when it is installed:
  * scores_kernel  -- auction scores with linear interpolation in the belief (backward pass);
  * gibbs_kernel   -- softmax policy and log-sum-exp value (backward pass);
  * jump_scatter   -- redistribution of mass after an auction epoch (forward pass).
Without numba, or with the environment variable NO_NUMBA=1, the original numpy code in
verify_auction.py and calib_model.py runs instead.  Both paths give the same numbers up to
floating-point summation order (differences of order 1e-16).

These kernels serve only the floating-point numerics.  The interval certificates
(cert_interval.py, cert_general.py, cert_calib.py) use python-flint/Arb and never call them.
"""
import os
import math
import numpy as np

HAVE_NUMBA = False
if os.environ.get('NO_NUMBA', '0') != '1':
    try:
        import numba
        HAVE_NUMBA = True
    except ImportError:
        pass

if HAVE_NUMBA:
    njit = numba.njit(cache=True, fastmath=False)

    @njit
    def _interp_row(vrow, bel, h, npg, p):
        jf = int(math.floor(bel / h))
        if jf > npg - 2:
            jf = npg - 2
        fr = (bel - p[jf]) / h
        if fr < 0.0:
            fr = 0.0
        elif fr > 1.0:
            fr = 1.0
        return (1.0 - fr) * vrow[jf] + fr * vrow[jf + 1]

    @njit
    def scores_kernel(v, pw, g, bw, bl, h, p, nact):
        """score[ib, j, ia] = g + pw*V(b-a, Psi_1) + (1-pw)*V(b, Psi_0); -1e6 marks infeasible bids."""
        nb, npg = v.shape
        score = np.full((nb, npg, nact), -1e6)
        for ib in range(nb):
            amax = ib if ib < 2 else 2
            for ia in range(amax + 1):
                if ia == 0:
                    for j in range(npg):
                        score[ib, j, 0] = v[ib, j]
                    continue
                for j in range(npg):
                    vw = _interp_row(v[ib - ia], bw[j, ia], h, npg, p)
                    vl = _interp_row(v[ib], bl[j, ia], h, npg, p)
                    score[ib, j, ia] = g[j, ia] + pw[j, ia] * vw + (1.0 - pw[j, ia]) * vl
        return score

    @njit
    def gibbs_kernel(z):
        """Max-shifted log-sum-exp over the last axis and the Gibbs probabilities exp(z - lse)."""
        nb, npg, na = z.shape
        lse = np.empty((nb, npg)); pol = np.empty_like(z)
        for i in range(nb):
            for j in range(npg):
                mx = z[i, j, 0]
                for a in range(1, na):
                    if z[i, j, a] > mx:
                        mx = z[i, j, a]
                s = 0.0
                for a in range(na):
                    s += math.exp(z[i, j, a] - mx)
                l = mx + math.log(s); lse[i, j] = l
                for a in range(na):
                    pol[i, j, a] = math.exp(z[i, j, a] - l)
        return pol, lse

    @njit
    def jump_scatter(nextm, m, pol_t, pw, bw, bl, jump, h, p):
        """Add to nextm the mass that reaches each (budget, belief) node after an auction epoch."""
        nb, npg = m.shape
        for ib in range(nb):
            amax = ib if ib < 2 else 2
            for ia in range(amax + 1):
                for j in range(npg):
                    amt = jump * m[ib, j] * pol_t[ib, j, ia]
                    if ia == 0:
                        nextm[ib, j] += amt
                        continue
                    for k in range(2):
                        if k == 0:
                            prob = pw[j, ia]; bel = bw[j, ia]; newb = ib - ia
                        else:
                            prob = 1.0 - pw[j, ia]; bel = bl[j, ia]; newb = ib
                        src = amt * prob
                        jf = int(math.floor(bel / h))
                        if jf > npg - 2:
                            jf = npg - 2
                        fr = (bel - p[jf]) / h
                        if fr < 0.0:
                            fr = 0.0
                        elif fr > 1.0:
                            fr = 1.0
                        nextm[newb, jf] += src * (1.0 - fr)
                        nextm[newb, jf + 1] += src * fr
        return nextm

