"""Criteo uplift v2.1: split users into two regimes by the cross-fitted predicted effect of WINNING
(tau_W(x) = ITT(x)/exposure(x)); per regime estimate exposure (win) rate, AIPW-Wald effect of winning,
baseline visit rates of won and lost users. Used to calibrate (p, Delta_s levels, d_s ordering)."""
import numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
d=np.load(sys.argv[1]); X=d['X']; Z=d['Z'].astype(int); D=d['D'].astype(float); Y=d['Y'].astype(float)
n=len(Z); rng=np.random.default_rng(3); fold=rng.integers(0,2,n)
zeta=np.zeros(n); mD1=np.zeros(n); m1=np.zeros(n); m0=np.zeros(n)
def fit(idx,y,cap=2_000_000):
    idx=rng.choice(idx,min(cap,len(idx)),replace=False); return HGB(max_iter=200,random_state=0).fit(X[idx],y[idx])
for k in range(2):
    tr,te=np.where(fold!=k)[0],np.where(fold==k)[0]
    zeta[te]=fit(tr,Z).predict_proba(X[te])[:,1]
    t1=tr[Z[tr]==1]; t0=tr[Z[tr]==0]
    mD1[te]=fit(t1,D).predict_proba(X[te])[:,1]
    m1[te]=fit(t1,Y).predict_proba(X[te])[:,1]; m0[te]=fit(t0,Y,cap=len(t0)).predict_proba(X[te])[:,1]
z=np.clip(zeta,.02,.98)
tauW=(m1-m0)/np.maximum(mD1,1e-3)
psiY=m1-m0+Z*(Y-m1)/z-(1-Z)*(Y-m0)/(1-z); psiD=mD1+Z*(D-mD1)/z
# baseline of won users: E[Y(0)|D=1] via AIPW: E[Y D | Z=1] - ITT... use complier-mean formula
psiYD1=mD1*0+Z*(Y*D)/z            # E[Y D | Z=1] = E[Y(1) D(1)]
def wald(s):
    A,B=psiY[s].mean(),psiD[s].mean(); th=A/B; IF=(psiY[s]-th*psiD[s])/B
    return float(th),float(IF.std()/np.sqrt(s.sum()))
out={'n':int(n)}
for name,thr in (('median',np.median(tauW)),):
    hi=tauW>thr
    for lab,s in (('low',~hi),('high',hi)):
        th,se=wald(s)
        e=float(psiD[s].mean())                       # exposure (win) rate under treatment
        yd=float(psiYD1[s].mean())                    # E[Y(1) D]
        base_won=(yd-th*e)/e                          # E[Y(0)|won] = E[Y(1)|won]-effect
        y0=float(((1-Z)*Y/(1-z))[s].mean())           # E[Y(0)] overall in regime
        base_lost=(y0-base_won*e)/(1-e)
        out[f'{name}_{lab}']=dict(share=float(s.mean()),exposure=e,wald=th,se=se,itt=float(psiY[s].mean()),
                                  base_won=base_won,base_lost=base_lost,mean_tauW_pred=float(tauW[s].mean()))
th,se=wald(np.ones(n,bool)); out['pooled']=dict(wald=th,se=se,exposure=float(psiD.mean()))
# bootstrap (over users, multiplier) for regime quantities
B=200; boot=[]
for b in range(B):
    wgt=rng.exponential(size=n)
    r={}
    for lab,s in (('low',tauW<=np.median(tauW)),('high',tauW>np.median(tauW))):
        ww=wgt[s]; A=(ww*psiY[s]).sum()/ww.sum(); Bd=(ww*psiD[s]).sum()/ww.sum(); r[lab]=(A/Bd,Bd)
    boot.append([r['low'][0],r['high'][0],r['low'][1],r['high'][1]])
boot=np.array(boot); out['boot_sd']=dict(zip(['wald_low','wald_high','exp_low','exp_high'],boot.std(0).tolist()))
out['boot_prob_exp_high_lt_low']=float((boot[:,3]<boot[:,2]).mean())
np.save('criteo_tauW.npy',np.c_[tauW,mD1,z])
json.dump(out,open('criteo_regimes.json','w'),indent=1); print(json.dumps(out,indent=1))
