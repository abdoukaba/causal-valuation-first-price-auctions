"""Re-run the data calibration with the floor-censored rival law from floor_censored.py: refit the score scale
zeta to the censored win curve (same least-squares fit as calib_season1.py, top point 3h = 450), recompute the
Arb bound of Table 4 at the corrected law, and re-solve the calibrated market (reference choice, logit and
anchored profit ratios, selection gaps).  Variant B also replaces the click-rate gradient by the one measured on
wins where M is exactly observed.  Inputs: floor_censored.json, floor_censored.npz and the calib_full inputs.
Writes floor_censored_calib.json.  Floating-point numerics except the Arb bounds."""
import json, numpy as np
from scipy.optimize import minimize_scalar
import calib_full as cf, cert_general as cg
D = json.load(open('floor_censored.json')); Z = np.load('floor_censored.npz')
F = lambda x: 1 / (1 + np.exp(-x))
mg = np.array([0, 150, 300, 450]) / 3000
out = dict(inputs={k: D[k] for k in ('q0_censored', 'q0_naive', 'ctr_ratio_uncensored', 'ctr_ratio_all_wins', 'floor_binds_won')})
# censored CDF on the same 5-unit grid used by calib_season1.py is not stored; refit on the stored grid for both laws
grid = Z['grid']; Fc = Z['F']
def fitz(q0, emp):
    W = lambda a, z: sum(q0[j] * F((a - mg[j]) / z) for j in range(4))
    r = minimize_scalar(lambda lz: np.mean((W(grid / 3000, np.exp(lz)) - emp) ** 2), bounds=(np.log(.003), np.log(.5)), method='bounded')
    z = float(np.exp(r.x)); return z, float(np.sqrt(r.fun)), [float(W(.05, z)), float(W(.1, z))]
qn = np.array(D['q0_naive']); qc = np.array(D['q0_censored'])
zn, rn, wn = fitz(qn, Z['Fnaive'])
zc, rc, wc = fitz(qc, Fc)
out['zeta_naive'] = dict(zeta=zn, rmse=rn, W_bids=wn); out['zeta_censored'] = dict(zeta=zc, rmse=rc, W_bids=wc)
print(out, flush=True)
# Arb bounds (Table 4 analogue) at the censored law
cert = {}
qs = [str(round(x, 6)) for x in qc[:-1]]; qs.append(str(round(1 - sum(float(x) for x in qs), 6)))
for zeta in (round(zc, 4), 0.1, 0.2):
    for eta in ('0.3', '0.45'):
        best = None
        try:
            for th in ('4', '8', '16', '32'):
                k = cg.kappa(mg.round(6).tolist(), qs, str(zeta), 1, '2.25', '0.05', eta, th)
                u = k['kappa'].upper(); s = u.str(3, radius=False) if hasattr(u, 'str') else str(u)
                if best is None or float(u) < best[0] or (best[0] == float('inf') and s < best[2]):
                    best = (float(u), th, s, bool(k['certified']))
            cert[f'zeta={zeta},eta={eta}'] = dict(kappa_upper=best[2], theta=best[1], certified=best[3])
        except ValueError as e: cert[f'zeta={zeta},eta={eta}'] = str(e)
        print(zeta, eta, cert.get(f'zeta={zeta},eta={eta}'), flush=True)
out['arb'] = cert
# calibrated market
ctr_all = np.array([r['ctr'] for r in cf.CT['ctr_by_bin']]); phi_all = np.r_[ctr_all, ctr_all[-1]]
cu = D['ctr_uncensored']; ctr_u = np.array([cu['0'], cu['h'], cu['2h']]); phi_u = np.r_[ctr_u, ctr_u[-1]]
def ev(tag, q0, zeta, phi_raw):
    P = cf.primitives(q0=q0, zeta=zeta, phi=phi_raw / (q0 @ phi_raw)); r = cf.single_auction_profit(P); a = int(np.argmax(r))
    e0 = cf.equilibrium(P, None, seeds=3); ref = np.full(3, .05); ref[a] = .9; e1 = cf.equilibrium(P, ref, seeds=3)
    sel = cf.selection_split(P)
    out[tag] = dict(q0=q0.tolist(), zeta=zeta, d1=float(P['d'][1]), profits=r.tolist(), selected=a, logit=e0['ratio'],
                    anchored=e1['ratio'], spread=max(e0['spread'], e1['spread']),
                    gap_h=float(sel['0.05']['gap']), gap_2h=float(sel['0.1']['gap']))
    print(tag, out[tag], flush=True)
ev('base (paper)', cf.Q0, cf.ZETA, phi_all)
ev('A: censored q0, refit zeta', qc, zc, phi_all)
ev('B: A + click gradient on uncensored wins', qc, zc, phi_u)
json.dump(out, open('floor_censored_calib.json', 'w'), indent=1, default=float)
