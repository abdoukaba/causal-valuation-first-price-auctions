"""Section 9.1-9.2: sensitivity of the calibrated market to floor censoring in the iPinYou win curve.
In this second-price market a win pays max(M, floor); on the 23.4% of wins where the floor binds, the true highest
rival bid M lies at or below the recorded price, so the estimated background law q0 overstates high bids.
Without the raw log we bound the effect: move the floor-bound share of won mass (0.234 x win rate = 0.1586) from the
recorded bins down to the lowest bin (worst case), and half of it. The score scale zeta is kept at its fitted value.
Writes floor_sensitivity.json. Floating-point numerics."""
import numpy as np, json, calib_full as cf
out = {}
q = cf.Q0.copy(); m = 0.2341818624659281 * q[:3].sum()
def ev(q2, tag):
    P = cf.primitives(q0=q2); r = cf.single_auction_profit(P); a = int(np.argmax(r))
    e0 = cf.equilibrium(P, None, seeds=3); ref = np.full(3, .05); ref[a] = .9; e1 = cf.equilibrium(P, ref, seeds=3)
    sel = cf.selection_split(P)
    out[tag] = dict(q0=q2.tolist(), d1=float(P['d'][1]), profits=r.tolist(), selected=a, logit=e0['ratio'],
                    anchored=e1['ratio'], spread=max(e0['spread'], e1['spread']),
                    gap_h=float(sel['0.05']['gap']), gap_2h=float(sel['0.1']['gap']))
    print(tag, out[tag], flush=True)
ev(q, 'base')
qa = q.copy(); fr = q[1:3] / q[1:3].sum(); qa[1:3] -= m * fr; qa[0] += m; ev(qa, 'proportional from h,2h to 0')
qb = q.copy(); qb[1] -= m; qb[0] += m; ev(qb, 'all from h to 0')
qc = q.copy(); qc[1] -= m / 2; qc[0] += m / 2; ev(qc, 'half of the floor-bound mass')
json.dump(out, open('floor_sensitivity.json', 'w'), indent=1)
