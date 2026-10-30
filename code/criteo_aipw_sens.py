"""Cross-fitted AIPW Wald estimator of the effect of winning (exposure) on visit/conversion.
Allows treatment assignment to depend on X (Criteo v2.1 pools tests with different treatment
ratios). theta = E[psi_Y]/E[psi_D], psi_V = m1-m0 + Z(V-m1)/zeta - (1-Z)(V-m0)/(1-zeta)."""
import numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
d=np.load(sys.argv[1]); X=d['X']; Z=d['Z'].astype(int); D=d['D'].astype(float)
Ys={'visit':d['Y'].astype(float),'conversion':d['C'].astype(float)}
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 1; K=int(sys.argv[3]) if len(sys.argv)>3 else 2; TRIM=float(sys.argv[4]) if len(sys.argv)>4 else .02
rng=np.random.default_rng(SEED); fold=rng.integers(0,K,len(Z))
n=len(Z); zeta=np.zeros(n); mD1=np.zeros(n); mY={k:[np.zeros(n),np.zeros(n)] for k in Ys}
def fit(idx,y,cap=2_000_000):
    idx=rng.choice(idx,min(cap,len(idx)),replace=False)
    return HGB(max_iter=150,random_state=0).fit(X[idx],y[idx])
aucs=[]
for k in range(K):
    tr,te=np.where(fold!=k)[0],np.where(fold==k)[0]
    cz=fit(tr,Z); zeta[te]=cz.predict_proba(X[te])[:,1]; aucs.append(roc_auc_score(Z[te],zeta[te]))
    t1=tr[Z[tr]==1]; t0=tr[Z[tr]==0]
    mD1[te]=fit(t1,D).predict_proba(X[te])[:,1]
    for name,Y in Ys.items():
        mY[name][1][te]=fit(t1,Y).predict_proba(X[te])[:,1]
        mY[name][0][te]=fit(t0,Y,cap=len(t0)).predict_proba(X[te])[:,1]
zeta=np.clip(zeta,TRIM,1-TRIM)
out={'auc_Z_on_X':float(np.mean(aucs)),'zeta_range':[float(zeta.min()),float(zeta.max())],
     'trim_share':float(np.mean((zeta<=TRIM)|(zeta>=1-TRIM))),'seed':SEED,'K':K,'trim':TRIM}
psiD=mD1+Z*(D-mD1)/zeta   # m0 for D is 0 (one-sided)
cuts=np.quantile(mD1,[.5,.8,.9,.95]); strat=np.digitize(mD1,cuts)
for name,Y in Ys.items():
    m0,m1=mY[name][0],mY[name][1]
    psiY=m1-m0+Z*(Y-m1)/zeta-(1-Z)*(Y-m0)/(1-zeta)
    # baseline among losers in treated arm and won-baseline: E[Y0 D(1)] = E[Y0] - E[Y0 (1-D(1))]
    psiY0=m0+(1-Z)*(Y-m0)/(1-zeta)              # E[Y(0)]
    psiY0lost=(m1*0)+Z*(Y*(1-D))/zeta           # E[Y(0)(1-D(1))] (losers: Y=Y0)
    def ratio(a,b,s):
        A,B=a[s].mean(),b[s].mean(); th=A/B; IF=(a[s]-th*b[s])/B; return th,IF.std()/np.sqrt(s.sum())
    res={}
    allm=np.ones(n,bool)
    th,se=ratio(psiY,psiD,allm)
    pi=psiD.mean(); y0=psiY0.mean(); y0l=psiY0lost.mean()
    b_won=(y0-y0l)/pi; b_lost=y0l/(1-pi)
    res['overall']=dict(ATT_win=th,se=se,win_rate=pi,Y0_won=b_won,Y0_lost=b_lost,selection=b_won-b_lost,
                        naive_ITT=Y[Z==1].mean()-Y[Z==0].mean(),aipw_ITT=psiY.mean())
    res['strata']=[]
    for k in range(5):
        s=strat==k; th,se=ratio(psiY,psiD,s); pi=psiD[s].mean()
        y0=psiY0[s].mean(); y0l=psiY0lost[s].mean()
        res['strata'].append(dict(mean_e=float(mD1[s].mean()),win_rate=pi,ATT_win=th,se=se,
              Y0_won=(y0-y0l)/pi,Y0_lost=y0l/(1-pi)))
    out[name]=res
json.dump(out,open(f'criteo_aipw_s{SEED}_k{K}_t{TRIM}.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
