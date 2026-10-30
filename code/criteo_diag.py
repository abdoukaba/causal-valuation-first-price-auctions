"""Criteo diagnostics requested by referees: on the SAME trimmed sample, unadjusted / IPW-only /
outcome-regression-only / AIPW Wald estimates; covariate balance (standardized mean differences of the 12
features, raw vs IPW-weighted); propensity calibration by decile."""
import numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
d=np.load(sys.argv[1]); X=d['X']; Z=d['Z'].astype(int); D=d['D'].astype(float); Y=d['Y'].astype(float)
rng=np.random.default_rng(1); K=2; fold=rng.integers(0,K,len(Z)); n=len(Z)
zeta=np.zeros(n); mD1=np.zeros(n); m1=np.zeros(n); m0=np.zeros(n)
def fit(idx,y,cap=2_000_000):
    idx=rng.choice(idx,min(cap,len(idx)),replace=False); return HGB(max_iter=150,random_state=0).fit(X[idx],y[idx])
for k in range(K):
    tr,te=np.where(fold!=k)[0],np.where(fold==k)[0]
    zeta[te]=fit(tr,Z).predict_proba(X[te])[:,1]
    t1=tr[Z[tr]==1]; t0=tr[Z[tr]==0]
    mD1[te]=fit(t1,D).predict_proba(X[te])[:,1]
    m1[te]=fit(t1,Y).predict_proba(X[te])[:,1]; m0[te]=fit(t0,Y,cap=len(t0)).predict_proba(X[te])[:,1]
keep=(zeta>=.02)&(zeta<=.98); z=np.clip(zeta,.02,.98)
def ratio(a,b,s):
    A,B=a[s].mean(),b[s].mean(); th=A/B; IF=(a[s]-th*b[s])/B; return float(th),float(IF.std()/np.sqrt(s.sum()))
out={'n':int(n),'kept':int(keep.sum()),'auc':float(roc_auc_score(Z,zeta))}
# unadjusted (difference in means / win rate) on same sample, as an IF ratio
p1=Z.mean()
psiY_un=Z*Y/Z[keep].mean()-(1-Z)*Y/(1-Z[keep].mean()); psiD_un=Z*D/Z[keep].mean()
psiY_ipw=Z*Y/z-(1-Z)*Y/(1-z); psiD_ipw=Z*D/z
psiY_ra=m1-m0; psiD_ra=mD1
psiY_aipw=m1-m0+Z*(Y-m1)/z-(1-Z)*(Y-m0)/(1-z); psiD_aipw=mD1+Z*(D-mD1)/z
for name,(a,b) in {'unadjusted':(psiY_un,psiD_un),'ipw':(psiY_ipw,psiD_ipw),'regression':(psiY_ra,psiD_ra),'aipw':(psiY_aipw,psiD_aipw)}.items():
    out[name]=dict(zip(('effect','se'),ratio(a,b,keep))); out[name]['itt']=float(a[keep].mean())
# balance
w=np.where(Z==1,1/z,1/(1-z))
smd_raw=[];smd_w=[]
for j in range(X.shape[1]):
    x=X[keep,j].astype(float); zz=Z[keep]; ww=w[keep]
    s=np.sqrt((x[zz==1].var()+x[zz==0].var())/2)+1e-12
    smd_raw.append(float((x[zz==1].mean()-x[zz==0].mean())/s))
    smd_w.append(float((np.average(x[zz==1],weights=ww[zz==1])-np.average(x[zz==0],weights=ww[zz==0]))/s))
out['smd_raw']=smd_raw; out['smd_weighted']=smd_w
# calibration by decile of fitted propensity
q=np.quantile(zeta,np.linspace(0,1,11)); b=np.clip(np.searchsorted(q,zeta,side='right')-1,0,9)
out['calibration']=[dict(decile=k,mean_pred=float(zeta[b==k].mean()),obs_rate=float(Z[b==k].mean()),
                         visit_ctrl=float(Y[(b==k)&(Z==0)].mean()),visit_treat=float(Y[(b==k)&(Z==1)].mean())) for k in range(10)]
json.dump(out,open('criteo_diag.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k not in('calibration',)},indent=1))
for r in out['calibration']: print(r)
