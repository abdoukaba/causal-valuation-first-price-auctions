"""Theorem 3 (v4.0) constants: e_N = e^{BT}[3(eps_N+2lamT/N)+4(1+BT)/(N-1)], B=1.5*lam*eta*omega_bar.
No parameter-specific absorption step."""
import math, json, numpy as np, grid_certificate as gc
def consts(T, lam, eps, eta, gap_num, Hent):
    m = gc.Model(11, 4000, eta, eps, T, lam, check=False)
    r = gc.best_continuum(m)
    pc, LPsi, rmax = gc.primitive_constants(m)
    Lw = max(c['Lw'] for c in pc.values()); om = max(c['omega'] for c in pc.values()); gam = max(c['gamma'] for c in pc.values())
    LipPi = r['LipPi0']; sig = m.sigma
    Theta = (LipPi + lam*T*(Lw + LipPi/2))*math.exp(lam*(LPsi-1)*T + sig**2*T)
    B = 1.5*lam*eta*om
    prof = m.value[:, None, :] - m.actions[1:, None][None]
    imm = float(np.abs(prof).max()); osc = (m.nb-1)*(max(prof.max(), 0)-min(prof.min(), 0))
    rows = []
    for N in (10**2, 10**3, 10**4, 10**6):
        epsN = math.sqrt((0.25 + lam*T + Theta**2*sig**2*T/16)/N)
        e = math.exp(B*T)*(3*(epsN + 2*lam*T/N) + 4*(1+B*T)/(N-1))
        mf = lam*T*eta*(om*(imm+osc)+gam)*e
        rows.append(dict(N=N, eN=e, mf=mf, generic=eps*lam*T*math.log(3)+mf, envelope=Hent+mf, apost=gap_num+mf))
    return dict(Theta=Theta, BT=B*T, omega=om, gamma=gam, imm=imm, osc=osc, rows=rows)
out = {'short': consts(.2, .1, .5, .5, 2.628618176256312e-4, 0.010986),
       'orig045': consts(1., 2.25, .05, .45, 0.02397, 0.117179)}
json.dump(out, open('finite_n_v4.json', 'w'), indent=1)
for k, v in out.items():
    print(k, {x: round(v[x], 5) for x in ('Theta', 'BT', 'omega', 'gamma', 'imm', 'osc')})
    for r in v['rows']: print('  ', {a: (round(b, 5) if isinstance(b, float) else b) for a, b in r.items()})
