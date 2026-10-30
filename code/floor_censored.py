"""Floor-censored rival-bid law from the raw iPinYou 2013-03-16 logs (bid and impression files).
On a win the recorded price is max(M, floor): if price > floor, M = price; if price <= floor, only M <= floor is known
(left censoring).  A loss at the fixed bid 300 means M >= 300.  Assuming the floor is independent of M, the
distribution of M below 300 is estimated by Kaplan-Meier in reversed time (left-censored -> right-censored).
Also checks the join (duplicate impression IDs) and recomputes the click-rate ratios by rival-bid bin on
uncensored wins.  Usage: python floor_censored.py bid.bz2 imp.bz2 clk.bz2   -> floor_censored.json, floor_censored.npz"""
import sys, json, numpy as np, pandas as pd
fb, fi, fc = sys.argv[1:4]
rd = lambda f, cols, names: pd.read_csv(f, sep='\t', header=None, quoting=3, usecols=cols, dtype=str, compression='bz2').set_axis(names, axis=1)
b = rd(fb, [0, 7, 14, 15, 16, 18], ['id', 'ex', 'vis', 'fmt', 'floor', 'bid'])
i = rd(fi, [0, 20], ['id', 'pay']); c = rd(fc, [0], ['id'])
out = dict(n_bids=len(b), n_imp_rows=len(i), dup_bid_ids=int(b.id.duplicated().sum()),
           dup_imp_rows=int(i.id.duplicated().sum()), imp_not_in_bids=int((~i.id.isin(b.id)).sum()))
d = b.merge(i, on='id', how='left'); out['joined_rows'] = len(d)
dd = d.drop_duplicates('id').copy()                        # one row per auction
dd['floor'] = dd.floor.astype(float); dd['bid'] = dd.bid.astype(float); dd['pay'] = dd.pay.astype(float)
dd['won'] = dd.pay.notna(); dd['clk'] = dd.id.isin(set(c.id))
out['bid_values'] = dd.bid.value_counts().head(3).to_dict()
w = dd[dd.won]; out['n_wins'] = int(len(w)); out['win_rate'] = float(dd.won.mean())
out['win_rate_joined_frame'] = float(d.pay.notna().mean())
bind = w.pay <= w.floor; out['floor_binds_won'] = float(bind.mean())
out['floor_bound_floor_quantiles'] = dict(zip(['p10', 'p25', 'p50', 'p75', 'p90', 'max'], np.percentile(w.floor[bind], [10, 25, 50, 75, 90, 100]).tolist()))
out['floor_bound_share_floor_ge'] = {str(t): float((w.floor[bind] >= t).mean()) for t in (25, 50, 75, 150, 225)}
out['loss_with_floor_above_bid'] = int(((~dd.won) & (dd.floor > dd.bid)).sum())
# --- reversed-time Kaplan-Meier for F(a) = P(M < a), a <= 300 ---------------------------------------------
n = len(dd); n_loss = int((~dd.won).sum())
x = np.where(bind, w.floor, w.pay).astype(float); ev = (~bind).to_numpy()   # event: M exactly observed
# reverse time T = 300 - M; losses (M >= 300) are events at T <= 0, i.e. the earliest; they leave the risk set first
order = np.argsort(-x, kind='stable'); x, ev = x[order], ev[order]
vals = np.unique(x)[::-1]                                  # descending M
at_risk = n - n_loss                                       # units with M < 300 not yet resolved
S = (n - n_loss) / n                                       # P(M < 300)
surv = {}; pos = 0
cnt_e = pd.Series(ev).groupby(x).sum(); cnt_all = pd.Series(np.ones_like(x)).groupby(x).sum()
F_at = {}
for v in vals:                                             # S(v-) bookkeeping: P(M < v) after removing events at v
    e = cnt_e[v]; tot = cnt_all[v]
    F_at[v] = S                                            # P(M <= v) before removing  (= P(M < v) + P(M = v))
    if e > 0: S *= 1 - e / at_risk
    at_risk -= tot
    surv[v] = S                                            # P(M < v)
vs = np.array(sorted(surv)); Fs = np.array([surv[v] for v in vs])
def F(a):                                                  # P(M < a) = surv at smallest observed v >= a
    j = np.searchsorted(vs, a, side='left')
    return float(Fs[j]) if j < len(vs) else float((n - n_loss) / n)
naive = lambda a: float(out['win_rate'] * (w.pay < a).mean())
grid = list(range(5, 301, 5))                              # the grid of calib_season1.py
out['cdf_censored'] = {a: F(a) for a in grid}; out['cdf_naive'] = {a: naive(a) for a in grid}
wr = out['win_rate']
for tag, f in (('censored', F), ('naive', naive)):
    out[f'q0_{tag}'] = [f(75), f(225) - f(75), wr - f(225), 1 - wr]
# --- click rate by rival-bid bin on uncensored wins (M exactly observed) --------------------------------------
u = w[~bind]; bins = pd.cut(u.pay, [-1, 75, 225, 300], right=False, labels=['0', 'h', '2h'])
ctr = u.groupby(bins, observed=False).clk.mean(); out['ctr_uncensored'] = ctr.to_dict()
out['ctr_ratio_uncensored'] = (ctr / ctr.iloc[0]).round(4).to_dict()
allb = pd.cut(w.pay, [-1, 75, 225, 300], right=False, labels=['0', 'h', '2h']); ca = w.groupby(allb, observed=False).clk.mean()
out['ctr_ratio_all_wins'] = (ca / ca.iloc[0]).round(4).to_dict()
json.dump(out, open('floor_censored.json', 'w'), indent=1, default=float)
np.savez_compressed('floor_censored.npz', grid=np.array(grid, float), F=np.array([F(a) for a in grid]), Fnaive=np.array([naive(a) for a in grid]))
print(json.dumps(out, indent=1, default=float))
