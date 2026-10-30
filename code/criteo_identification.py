"""Causal value of winning an ad auction from censored real logs (Criteo Uplift v2.1).
Z = randomized eligibility (treatment), D = exposure (won & displayed; censored: never observed
in control), Y = visit / conversion observed for everyone.  One-sided noncompliance:
 ATT_win = (E[Y|Z=1]-E[Y|Z=0]) / P(D=1|Z=1);  E[Y0|win] = (E[Y|Z=0]-(1-pi)E[Y|Z=1,D=0])/pi.
Heterogeneity by cross-fitted win-propensity e(X)=P(D=1|Z=1,X) (pre-treatment)."""
import numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingClassifier
d=np.load(sys.argv[1]); X,Z,D=d['X'],d['Z'].astype(bool),d['D'].astype(float)
assert D[~Z].sum()==0
def wald(Y,D,Z):
    y1,y0=Y[Z],Y[~Z]; d1=D[Z]; n1,n0=len(y1),len(y0)
    pi=d1.mean(); th=(y1.mean()-y0.mean())/pi
    var=(np.var(y1-th*d1)/n1+np.var(y0)/n0)/pi**2
    los=Y[Z&(D==0)].mean()
    b1=(y0.mean()-(1-pi)*los)/pi
    # variance of b1 via IF approx (delta method, numeric)
    yz0=y0.mean(); m10=(y1*(1-d1)).mean()
    b1=(yz0-m10)/pi   # identical to formula above
    g=np.array([1/pi,-1/pi,-(yz0-m10)/pi**2])   # wrt (E[Y|Z0], E[Y(1-D)|Z1], pi)
    C1=np.cov(np.vstack([y1*(1-d1),d1]))/n1
    vb=g[0]**2*np.var(y0)/n0+g[1:]@C1@g[1:]
    return dict(n_treat=int(n1),n_ctrl=int(n0),win_rate=pi,ITT=y1.mean()-y0.mean(),
      ATT_win=th,se_ATT=var**.5,base_Y0_won=b1,se_base_won=vb**.5,base_Y0_lost=los,
      naive_won_minus_lost=Y[Z&(D==1)].mean()-los,selection_bias=b1-los,ctrl_mean=y0.mean())
out={}
rng=np.random.default_rng(0); fold=rng.integers(0,2,len(Z)).astype(bool)
e=np.zeros(len(Z),dtype=np.float32)
for f in (False,True):
    tr=np.where((fold==f)&Z)[0]; sub=rng.choice(tr,min(3_000_000,len(tr)),replace=False)
    clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=.1,random_state=0).fit(X[sub],D[sub])
    te=np.where(fold!=f)[0]; e[te]=clf.predict_proba(X[te])[:,1]
cuts=np.quantile(e,[.5,.8,.9,.95]); strat=np.digitize(e,cuts)
for name in ('visit','conversion'):
    Y=d['Y' if name=='visit' else 'C'].astype(float)
    res=dict(overall=wald(Y,D,Z),strata=[])
    for k in range(5):
        s=strat==k; r=wald(Y[s],D[s],Z[s]); r['e_range']=[float(e[s].min()),float(e[s].max())]; r['mean_e']=float(e[s].mean())
        res['strata'].append(r)
    out[name]=res
# balance check: randomization -> e(X) distribution equal across arms
out['balance_mean_e']={'treat':float(e[Z].mean()),'ctrl':float(e[~Z].mean())}
json.dump(out,open('criteo_results.json','w'),indent=1,default=float)
for name in ('visit','conversion'):
    o=out[name]['overall']; print(name,{k:round(v,5) for k,v in o.items()})
    for r in out[name]['strata']: print('  e~%.4f win=%.4f ATT=%.4f(%.4f) Y0won=%.4f Y0lost=%.4f'%(r['mean_e'],r['win_rate'],r['ATT_win'],r['se_ATT'],r['base_Y0_won'],r['base_Y0_lost']))
print(out['balance_mean_e'])
