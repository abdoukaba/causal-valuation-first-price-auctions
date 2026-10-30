"""Composite data calibration (iPinYou rival law, win-curve sharpness and value-competition gradient;
Criteo incrementality regimes) + data-learned KL reference; equilibrium numerics and Arb bound."""
import json, sys, numpy as np
from scipy.optimize import brentq
from scipy.special import expit
import calib_model as cm, cert_calib as cc
R=json.load(open('criteo_regimes.json')); C=json.load(open('calib_season1.json'))['top=450']
CT=json.load(open('ctr_bins.json'))
MG=np.array([0,.05,.10,.15]); Q0=np.array(C['q0']); Q0=Q0/Q0.sum(); ZETA=C['zeta_fit']
lo,hi=R['median_low'],R['median_high']
P0=hi['share']; MEAN_DELTA=0.0793                     # normalization: same average value as Table 1
ctr=[r['ctr'] for r in CT['ctr_by_bin']]; PHI=np.array(ctr+[ctr[-1]]); PHI=PHI/(Q0@PHI)
def primitives(wald=(lo['wald'],hi['wald']),expo=(lo['exposure'],hi['exposure']),phi=PHI,zeta=ZETA,scale=1.0,q0=Q0,p0=P0,d_fixed=None):
    wald=np.array(wald); base=wald/(np.array([1-p0,p0])@wald)*MEAN_DELTA*scale
    Delta=np.outer(base,phi)
    # offsets: model win-rate ratio at the deployed bid 2h must equal the exposure ratio (regime 1 = high incrementality)
    def wbar(dd): return q0@expit((.10-MG-dd)/zeta)
    target=expo[1]/expo[0]
    f=lambda g: wbar(g/2)/wbar(-g/2)-target
    if d_fixed is None:
        g=brentq(f,-0.2,0.2); d=np.array([-g/2,g/2])
    else: d=np.array(d_fixed)
    return dict(Delta=Delta,d=d,zeta=zeta,q0=q0,p0=p0)
def single_auction_profit(P):
    out=[]
    for a in (.05,.10):
        out.append(sum([(1-P['p0']),P['p0']][s]*(P['q0']@(expit((a-MG-P['d'][s])/P['zeta'])*(P['Delta'][s]-a))) for s in range(2)))
    return np.array([0.]+out)
def model(P,eta=.45,eps=.05,rho=None,nb=21,nt=60):
    return cm.CalibModel(nb,nt,eta,eps,1.,2.25,MG,P['q0'],P['zeta'],P['d'],P['Delta'],P['p0'],rho)
def equilibrium(P,rho=None,eta=.45,eps=.05,seeds=6):
    sols=[]
    for sd in range(seeds):
        m=model(P,eta,eps,rho); s=m.solve(sd,max_iter=1500,damping=.4); sols.append((m,s))
    qs=np.array([s['q'] for _,s in sols]); spread=float(np.abs(qs-qs[0]).max())
    m,s=sols[0]; eq=m.monetary(s['q'],s['pol']); br=m.monetary(s['q'])
    return dict(eq=eq,br=br,ratio=eq/br if br else None,spread=spread,res=max(s['res'] for _,s in sols),q_mean=s['q'].mean(0).tolist())
def selection_split(P):
    out={}
    for a in (.05,.10):
        w=[expit((a-MG-P['d'][s])/P['zeta']) for s in (0,1)]; p=[1-P['p0'],P['p0']]; Q=P['q0']; D=P['Delta']
        W=sum(p[s]*Q@w[s] for s in (0,1))
        within=sum(p[s]*(Q@(D[s]*w[s])-(Q@D[s])*(Q@w[s])) for s in (0,1))/W
        between=p[0]*p[1]*(Q@D[1]-Q@D[0])*(Q@w[1]-Q@w[0])/W
        out[str(a)]=dict(gap=within+between,within=within,between=between,share=between/(within+between),winprob=W)
    return out
if __name__=='__main__':
    full={}
    for variant,dfix in (('criteo_offsets',None),('table1_offsets',[-.01,.01])):
        P=primitives(d_fixed=dfix); out={'primitives':{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in P.items()}}
        R1=single_auction_profit(P); out['single_auction_profit']=R1.tolist(); out['data_reference_bid']=int(np.argmax(R1))
        out['selection_split']=selection_split(P)
        # Sampling stability is assessed by real-record resampling in ref_learning_real.py.
        a=out['data_reference_bid']; refs={'uniform':None}
        for k in (0,1,2):
            r=np.full(3,.05); r[k]=.90; refs[f'ref{k}']=r
        refs['mild_data_ref']=np.where(np.arange(3)==a,.6,.2)
        out['equilibria']={}
        for name,rho in refs.items():
            out['equilibria'][name]=equilibrium(P,rho); print(variant,name,out['equilibria'][name],flush=True)
        out['arb']={}
        Ds=[[repr(float(x)) for x in row] for row in P['Delta']]; ds=[repr(float(x)) for x in P['d']]; qs=[repr(float(x)) for x in P['q0']]
        for lab,z in (('fitted',repr(float(P['zeta']))),('zeta0.2','0.2')):
            for rm in (None,'0.2','0.05'):
                try: out['arb'][f'{lab},rho_min={rm}']=cc.best(MG.tolist(),qs,Ds,ds,z,1,'2.25','0.05','0.45',rho_min=rm)
                except ValueError as e: out['arb'][f'{lab},rho_min={rm}']=str(e)
                print(variant,lab,rm,out['arb'][f'{lab},rho_min={rm}'],flush=True)
        full[variant]=out
        print(json.dumps({k:v for k,v in out.items() if k!='equilibria'},indent=1,default=float),flush=True)
    json.dump(full,open('calib_full.json','w'),indent=1,default=float)
