import json,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=json.load(open('season1_day.json')); C=json.load(open('calib_season1.json'))['top=450']
pay=np.load('pay_won.npy'); wr=S['win_rate']; a=np.arange(1,301)
emp=wr*np.searchsorted(np.sort(pay),a,side='left')/len(pay)
F=lambda x:1/(1+np.exp(-x)); mg=np.array([0,150,300,450])/3000; q=np.array(C['q0'])
W=lambda x,z,q,m: sum(q[j]*F((x-m[j])/z) for j in range(len(q)))
fig,ax=plt.subplots(1,2,figsize=(9,3.3))
ax[0].plot(a,emp,color='k',lw=2,label='iPinYou, 16 Mar 2013')
ax[0].plot(a,W(a/3000,C['zeta_fit'],q,mg),color='#e08a1e',label=f"model, data law, ζ = {C['zeta_fit']:.3f}")
ax[0].plot(a,W(a/3000,.2,q,mg),color='#2a6fdb',label='model, data law, ζ = 0.2')
ax[0].plot(a,W(a/3000,.2,np.array([.2,.4,.4]),mg[:3]),color='#2a6fdb',ls=':',label='model, Table 1')
ax[0].set_xlabel('own bid (logged bid = 300)'); ax[0].set_ylabel('win probability'); ax[0].set_ylim(0,1)
ax[0].set_title('Win curve (losses observed)',fontsize=9); ax[0].legend(frameon=False,fontsize=7,loc='upper left')
c=S['ctr_by_pay_decile']; ax[1].plot([r['pay'] for r in c],[r['ctr1000'] for r in c],'o-',color='#d9534f')
ax[1].set_xlabel('paying price = highest rival bid (decile median)'); ax[1].set_ylabel('clicks per 1,000 impressions')
ax[1].set_title('Value rises with competition',fontsize=9)
for x in ax:
    for s in ('top','right'): x.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('ipinyou.png',dpi=200)
