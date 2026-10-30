"""Robustness of the calibrated-market results (Section 9.2, 'Robustness to the data inputs'):
(a) common rescaling of both Criteo exposure effects (unadjusted/adjusted level gap) -> no change by construction;
(b) between-group effect ratio tau_1/tau_0 in {1.5, 2.0, 3.5};
(c) first-price shading proxy: move a share of the background mass at each bid level one step down.
Floating-point numerics; 3 Picard starts per equilibrium."""
import json, numpy as np, calib_full as cf
lo, hi = cf.R['median_low'], cf.R['median_high']; out = {}
def ev(P, tag):
    r = cf.single_auction_profit(P); a = int(np.argmax(r))
    e0 = cf.equilibrium(P, None, seeds=3); ref = np.full(3, .05); ref[a] = .9; e1 = cf.equilibrium(P, ref, seeds=3)
    out[tag] = dict(selected=a, logit_ratio=e0['ratio'], anchored_ratio=e1['ratio'], spread=max(e0['spread'], e1['spread']),
                    selection_gap_h=float(cf.selection_split(P)['0.05']['gap'])); print(tag, out[tag], flush=True)
ev(cf.primitives(), 'base')
s = 0.2870 / 0.2112; ev(cf.primitives(wald=(lo['wald'] * s, hi['wald'] * s)), 'level x1.36')
for rr in (1.5, 2.0, 3.5): ev(cf.primitives(wald=(hi['wald'] / rr, hi['wald'])), f'ratio {rr}')
for sh in (.25, .5):
    q = cf.Q0.copy(); q2 = q.copy()
    for j in (1, 2, 3): mv = sh * q[j]; q2[j] -= mv; q2[j - 1] += mv
    ev(cf.primitives(q0=q2), f'shade {sh}')
json.dump(out, open('sensitivity_inputs.json', 'w'), indent=1, default=float)
