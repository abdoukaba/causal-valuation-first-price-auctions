import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import grid_certificate as gc
out={}
fig,ax=plt.subplots(1,2,figsize=(9,3.2),sharey=True)
for k,(tag,args) in enumerate([('short',(41,48,.5,.5,.2,.1)),('orig045',(21,60,.45,.05,1.,2.25))]):
    m=gc.Model(*args); sol=m.solve(0,max_iter=800,damping=.5); q=sol['q']
    w=m.w[:,1:,:]; out[tag]=dict(dev_from_uniform=float(np.abs(q-1/3).max()),wmin=float(w.min()),wmax=float(w.max()),
        mon=m.monetary_policy_value(q,sol['pol']),br=m.monetary_policy_value(q,None),lamT=m.lam*m.T,res=sol['res'])
    t=np.arange(m.nt)*m.dt
    for a,c,l in zip(range(3),['#5b6b7a','#2a6fdb','#e08a1e'],['no bid','bid h','bid 2h']): ax[k].plot(t,q[:,a],color=c,label=l)
    ax[k].set_ylim(0,1); ax[k].axhline(1/3,color='0.7',lw=.8,ls=':'); ax[k].set_xlabel('time')
    ax[k].set_title(['Short horizon (λT = 0.02)','Long horizon, η = 0.45 (λT = 2.25)'][k],fontsize=9)
    for s in ('top','right'): ax[k].spines[s].set_visible(False)
ax[0].set_ylabel('population bid share'); ax[1].legend(frameon=False,fontsize=8,loc='upper right')
plt.tight_layout(); plt.savefig('equilibrium.png',dpi=200)
E=json.load(open('eps_table_arb.json')); T=json.load(open('eps_tradeoff.json'))
eps=[e['eps'] for e in T]
fig,ax=plt.subplots(1,2,figsize=(9,3.2))
for eta,c in (('0.45','#e08a1e'),('0.5','#2a6fdb')):
    ax[0].plot(eps,[E[f'{eta},{e}']['kappa_upper'] for e in eps],'o-',color=c,label=f'η = {eta}')
ax[0].axhline(1,color='0.4',ls='--',lw=1); ax[0].set_xlabel('entropy weight ε'); ax[0].set_ylabel('Arb upper bound on κ_θ'); ax[0].legend(frameon=False,fontsize=8)
ax[0].set_title('Contraction bound (verified below dashed line)',fontsize=9)
G=json.load(open('envelope_gap_045.json')); ge=sorted(float(k) for k in G); ax[1].plot(ge,[G[k]['J_BR']/G[k]['J_mon'] for k in sorted(G,key=float)],'s-',color='#d9534f'); ax[1].set_xlabel('entropy weight ε'); ax[1].set_ylabel('best-response / equilibrium profit')
ax[1].set_title('Exploitability (η = 0.45, numerical)',fontsize=9); ax[1].set_ylim(0,6.5)
for a in ax:
    for s in ('top','right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('tradeoff.png',dpi=200)
json.dump(out,open('v40_calib.json','w'),indent=1); print(json.dumps(out,indent=1))
