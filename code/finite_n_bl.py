"""Theorem S2 constants via the duality (bounded-Lipschitz) chaos argument: no e^{alpha T}
spatial-Lipschitz blow-up.  Uses LipPi from grid_certificate.continuum_kappa machinery."""
import math, json, numpy as np, grid_certificate as gc
def bl_constants(T, lam, eps, eta, gap_num):
    m = gc.Model(11, 4000, eta, eps, T, lam, check=False)
    r = gc.best_continuum(m)
    pc, LPsi, rmax = gc.primitive_constants(m)
    Lw = max(c['Lw'] for c in pc.values()); om = max(c['omega'] for c in pc.values()); gam = max(c['gamma'] for c in pc.values())
    LipPi = r['LipPi0']; J = 3; R = m.R; sig = m.sigma
    Theta = (LipPi + lam * T * (Lw + LipPi / 2)) * math.exp(lam * (LPsi - 1) * T) * math.exp(sig ** 2 * T)  # type-augmented drift
    grow = math.exp(1.5 * lam * eta * om * T)
    M0 = lam * T * R
    prof = m.value[:, None, :] - m.actions[1:, None][None]   # Delta_s(m)-a, a>0
    Bwins = (m.nb - 1)  # at most B/h_min wins
    imm = float(np.abs(prof).max()); osc = Bwins * (max(prof.max(), 0) - min(prof.min(), 0))
    rows = []
    for N in (10, 100, 1000, 10**4, 10**6):
        epsN = math.sqrt((0.25 + lam * T + Theta ** 2 * sig ** 2 * T / 16) / N)  # + initial fluctuation
        e = (3 * (epsN + 2 * lam * T / N) + 4 / (N - 1)) * grow
        mf = lam * T * eta * (om * (imm + osc) + gam) * e  # win-mismatch + expected-reward term
        rows.append(dict(N=N, rival_law_err=e, mf_term=mf, rigorous=eps * lam * T * math.log(J) + mf,
                         aposteriori=gap_num + mf))
    return dict(gamma=gam, imm=imm, osc=osc, LipPi0=LipPi, Theta=Theta, growth=grow, omega=om, kappa=r['kappa'], rows=rows)
out = {'certified_spec': bl_constants(.2, .1, .5, .5, 2.628618176256312e-4),
       'original_spec': bl_constants(1., 2.25, .05, .5, 0.02399356877816143)}
json.dump(out, open('finite_n_bl.json', 'w'), indent=1)
for k, v in out.items():
    print(k, {x: round(v[x], 4) for x in ('LipPi0', 'Theta', 'growth', 'kappa')})
    for r in v['rows']: print('  ', {a: (round(b, 6) if isinstance(b, float) else b) for a, b in r.items()})
