"""Journal figures (Springer style): vector PDF, embedded Arial-compatible font, panel labels (a)/(b), no titles
inside figures, line styles that survive grayscale.  Writes Fig1.pdf (equilibrium paths), Fig2.pdf (trade-off),
Fig3.pdf (iPinYou), ESM_Fig1.pdf (reference selection, Online Resource 1).
Inputs in the working directory: eps_table_arb.json, eps_tradeoff.json, envelope_gap_045.json, season1_day.json,
calib_season1.json, pay_won.npy, ref_learning_real.json."""
import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
                     'pdf.fonttype': 42, 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7, 'lines.linewidth': 1.2, 'axes.linewidth': 0.6, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'})
import grid_certificate as gc
W_MM = 174 / 25.4                                      # Springer double-column width
def label(ax, s): ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
def clean(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
STY = [dict(color='#4d4d4d', ls='-'), dict(color='#2a6fdb', ls='--'), dict(color='#e08a1e', ls='-.')]
# Fig 1: equilibrium bid shares
fig, ax = plt.subplots(1, 2, figsize=(W_MM, 2.5), sharey=True)
for k, args in enumerate([(41, 48, .5, .5, .2, .1), (21, 60, .45, .05, 1., 2.25)]):
    m = gc.Model(*args); sol = m.solve(0, max_iter=800, damping=.5); q = sol['q']; t = np.arange(m.nt) * m.dt
    for a, l in zip(range(3), ['no bid', 'bid $h$', 'bid $2h$']): ax[k].plot(t, q[:, a], label=l, **STY[a])
    ax[k].set_ylim(0, 1); ax[k].axhline(1 / 3, color='0.7', lw=.6, ls=':'); ax[k].set_xlabel('time'); clean(ax[k]); label(ax[k], '(a)(b)'[3*k:3*k+3])
ax[0].set_ylabel('population bid share'); ax[1].legend(frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig('Fig1.pdf'); plt.close(fig)
# Fig 2: contraction bound and exploitability against epsilon
E = json.load(open('eps_table_arb.json')); T = json.load(open('eps_tradeoff.json')); eps = [e['eps'] for e in T]
fig, ax = plt.subplots(1, 2, figsize=(W_MM, 2.5))
for (eta, st, mk) in (('0.45', STY[2], 'o'), ('0.5', STY[1], 's')):
    ax[0].plot(eps, [E[f'{eta},{e}']['kappa_upper'] for e in eps], marker=mk, ms=3.5, label=f'$\\eta$ = {eta}', **st)
ax[0].axhline(1, color='0.4', ls=':', lw=.8); ax[0].set_xlabel('entropy weight $\\varepsilon$'); ax[0].set_ylabel('Arb upper bound on $\\kappa_\\theta$')
ax[0].legend(frameon=False)
G = json.load(open('envelope_gap_045.json')); ks = sorted(G, key=float)
ax[1].plot([float(k) for k in ks], [G[k]['J_BR'] / G[k]['J_mon'] for k in ks], marker='s', ms=3.5, color='#b03a2e', ls='-')
ax[1].set_xlabel('entropy weight $\\varepsilon$'); ax[1].set_ylabel('best-response / equilibrium profit'); ax[1].set_ylim(0, 6.5)
for k in range(2): clean(ax[k]); label(ax[k], '(a)(b)'[3*k:3*k+3])
fig.tight_layout(); fig.savefig('Fig2.pdf'); plt.close(fig)
# Fig 3: iPinYou win curve and click rate by competition
S = json.load(open('season1_day.json')); C = json.load(open('calib_season1.json'))['top=450']
pay = np.load('pay_won.npy'); wr = S['win_rate']; a = np.arange(1, 301)
emp = wr * np.searchsorted(np.sort(pay), a, side='left') / len(pay)
F = lambda x: 1 / (1 + np.exp(-x)); mg = np.array([0, 150, 300, 450]) / 3000; q = np.array(C['q0'])
Wm = lambda x, z, q, m: sum(q[j] * F((x - m[j]) / z) for j in range(len(q)))
fig, ax = plt.subplots(1, 2, figsize=(W_MM, 2.6))
ax[0].plot(a, emp, color='k', lw=1.8, label='iPinYou, 16 Mar 2013')
ax[0].plot(a, Wm(a / 3000, C['zeta_fit'], q, mg), label=f"model, data law, $\\zeta$ = {C['zeta_fit']:.3f}", **STY[2])
ax[0].plot(a, Wm(a / 3000, .2, q, mg), label='model, data law, $\\zeta$ = 0.2', **STY[1])
ax[0].plot(a, Wm(a / 3000, .2, np.array([.2, .4, .4]), mg[:3]), color='#2a6fdb', ls=':', label='model, Table 1')
ax[0].set_xlabel('own bid (logged bid = 300)'); ax[0].set_ylabel('win probability'); ax[0].set_ylim(0, 1)
ax[0].legend(frameon=False, loc='upper left', fontsize=6.5)
c = S['ctr_by_pay_decile']; ax[1].plot([r['pay'] for r in c], [r['ctr1000'] for r in c], marker='o', ms=3.5, color='#b03a2e')
ax[1].set_xlabel('paying price = highest rival bid (decile median)'); ax[1].set_ylabel('clicks per 1,000 impressions')
for k in range(2): clean(ax[k]); label(ax[k], '(a)(b)'[3*k:3*k+3])
fig.tight_layout(); fig.savefig('Fig3.pdf'); plt.close(fig)
# Online Resource figure: reference selection under resampling
import audit_real_data as ar
rows = ar.summarize(json.load(open('ref_learning_real.json')))
x = np.array([r['n_ipinyou'] for r in rows]); un = np.array([r['p_h_all_attempts'] for r in rows]); iv = np.array([r['all_attempts_wilson95'] for r in rows])
fig, ax = plt.subplots(figsize=(W_MM * .8, 2.7))
ax.errorbar(x, 100 * un, yerr=100 * np.vstack((un - iv[:, 0], iv[:, 1] - un)), fmt='o-', ms=3.5, color='#205680', capsize=2.5, lw=1.2, label='select $h$ / all 100 draws')
ax.plot(x, 100 * np.array([r['p_h_conditional'] for r in rows]), 's--', ms=3.5, color='#ba6324', label='select $h$ / estimable draws')
ax.plot(x, [r['estimable'] for r in rows], '^:', ms=3.5, color='#555555', label='estimable / all 100 draws')
ax.set_xscale('log'); ax.set_ylim(-2, 104); ax.set_xlabel('iPinYou auctions resampled per draw (log scale)'); ax.set_ylabel('percentage of draws')
ax.set_xticks(x, [f'{v:,}' for v in x], rotation=25, ha='right'); ax.minorticks_off(); clean(ax); ax.legend(loc='lower right', frameon=False)
fig.tight_layout(); fig.savefig('ESM_Fig1.pdf'); plt.close(fig)
print('ok')
