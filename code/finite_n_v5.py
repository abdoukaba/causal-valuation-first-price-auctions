"""Theorem 2 constants (coupling proof, Appendix B):
e_N = e^{Ups T} [ 3/(2 sqrt(N-1)) + 2 Ups T/(N-1) ],  Ups = lambda eta omega_bar,
delta_N = (1/2) lambda T eta (gamma_bar + omega_bar O_bar) e_N,   G_MF - 2 delta_N <= G_N <= G_MF + 2 delta_N.
Floating-point evaluation of the stated formulas (Table 6)."""
import math, json, numpy as np, grid_certificate as gc
def consts(T, lam, eps, eta, gap_num):
    m = gc.Model(11, 4000, eta, eps, T, lam, check=False)
    pc, LPsi, rmax = gc.primitive_constants(m)
    om = max(c['omega'] for c in pc.values()); gam = max(c['gamma'] for c in pc.values())
    U = lam * eta * om
    prof = m.value[:, None, :] - m.actions[1:, None][None]
    Obar = (m.nb - 1) * (max(prof.max(), 0) - min(prof.min(), 0))
    Hbound = eps * lam * T * math.log(3); rows = []
    for N in (10**2, 10**3, 10**4, 10**6):
        e = math.exp(U * T) * (3 / (2 * math.sqrt(N - 1)) + 2 * U * T / (N - 1))
        two_delta = lam * T * eta * (gam + om * Obar) * e
        rows.append(dict(N=N, eN=e, two_delta=two_delta, upper_proved=Hbound + two_delta,
                         upper_numerical=gap_num + two_delta, lower_numerical=gap_num - two_delta))
    return dict(UpsT=U * T, omega=om, gamma=gam, Obar=Obar, G_MF_bound=Hbound, G_MF_numerical=gap_num, rows=rows)
out = {'short': consts(.2, .1, .5, .5, 2.628618176256312e-4), 'orig045': consts(1., 2.25, .05, .45, 0.02397)}
json.dump(out, open('finite_n_v5.json', 'w'), indent=1)
for k, v in out.items():
    print(k, {x: round(v[x], 5) for x in ('UpsT', 'omega', 'gamma', 'Obar', 'G_MF_bound')})
    for r in v['rows']: print('  ', {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()})
