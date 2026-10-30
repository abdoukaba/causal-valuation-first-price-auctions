"""Milgrom-Segal envelope identity for the monetary best-response gap at a fixed population flow q*:
V(s)=sup_pi [J_mon(pi)+s H(pi)] is convex in s, V'(s)=H(pi_s) (expected accumulated entropy), V(0)=J_BR.
Hence G(eps)=J_BR-J_mon(pi_eps)= eps V'(eps)-(V(eps)-V(0)) = int_0^eps (H_eps-H_s) ds <= eps*H_eps."""
import json, numpy as np, grid_certificate as gc
out = {}
for eps in (0.1, 0.05, 0.02, 0.01):
    m = gc.Model(21, 60, .5, eps, 1., 2.25); sol = m.solve(0, max_iter=600, damping=.5); q = sol['q']
    def V(s):
        if s == 0: return m.monetary_policy_value(q, None)
        m.entropy = s; v = m.backward(q)[0][0, -1, m.np // 2]; m.entropy = eps; return float(v)
    def Jmon(s):
        m.entropy = s; pol = m.backward(q)[1]; m.entropy = eps; return m.monetary_policy_value(q, pol)
    Ve, V0, Jm = V(eps), V(0), Jmon(eps)
    He = (Ve - Jm) / eps                       # exact: V(eps)=Jmon+eps*H at the optimum
    G_direct = V0 - Jm
    G_env = eps * He - (Ve - V0)               # identity, should equal G_direct
    ss = np.linspace(0, eps, 11)[1:]; Hs = [(V(s) - Jmon(s)) / s for s in ss]
    out[eps] = dict(J_BR=V0, J_mon=Jm, V_reg=Ve, H_eps=He, G_direct=G_direct, G_envelope=G_env,
                    bound_eps_H=eps * He, bound_eps_lamT_logJ=eps * 2.25 * np.log(3),
                    H_monotone=bool(np.all(np.diff(Hs) >= -1e-10)), H_path=[float(h) for h in Hs])
    print(eps, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out[eps].items() if k != 'H_path'}, flush=True)
json.dump(out, open('envelope_gap.json', 'w'), indent=1, default=float)
