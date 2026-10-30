"""Learning the anchoring reference from subsamples of the real data (replaces the synthetic EC.5 experiment).
For fraction f of each dataset: re-estimate (i) the iPinYou background law and click-rate profile from f of the
auctions and (ii) the Criteo regime effects and exposures (IPW-Wald, fixed regime split) from f of the users;
select the bid maximizing plug-in single-auction profit; record the selection and the equilibrium profit share
of the selected reference (looked up from the full-data equilibria)."""
import json, numpy as np, calib_full as cf
d=np.load('criteo.npz'); Z=d['Z'].astype(float); D=d['D'].astype(float); Y=d['Y'].astype(float)
tw=np.load('criteo_tauW.npy'); tau,z=tw[:,0],tw[:,2]; hi=tau>np.median(tau)
I=np.load('ipy_s1.npz'); won,pay,clk=I['won'],I['pay'],I['clk']
eqs=json.load(open('calib_full.json'))['criteo_offsets']['equilibria']; share={0:eqs['ref0']['ratio'],1:eqs['ref1']['ratio'],2:eqs['ref2']['ratio']}
rng=np.random.default_rng(7); nC,nI=len(Z),len(won); out={}
psiY=Z*Y/z-(1-Z)*Y/(1-z); psiD=Z*D/z
for f in (0.0003,0.001,0.003,0.01,0.03,0.1):
    sel=[]; fails=0
    for r in range(100):
        ic=rng.integers(0,nC,int(f*nC)); ii=rng.integers(0,nI,int(f*nI))
        h=hi[ic]; stats=[]
        for s in (~h,h):
            A=psiY[ic][s].mean(); B=psiD[ic][s].mean(); stats.append((A/B,B))
        w,p,c=won[ii],pay[ii],clk[ii]; wr=w.mean()
        F=lambda a: wr*np.mean(p[w]<a) if w.sum() else 0
        q0=np.array([F(75),F(225)-F(75),wr-F(225),1-wr])
        cb=[c[w&(p>=lo)&(p<up)].mean() if (w&(p>=lo)&(p<up)).sum() else np.nan for lo,up in ((0,75),(75,225),(225,300.1))]
        if np.any(np.isnan(cb)) or cb[0]==0 or min(stats[0][1],stats[1][1])<=0 or q0.min()<=0: fails+=1; continue
        phi=np.r_[cb,cb[-1]]; phi=phi/(q0@phi)
        try:
            P=cf.primitives((stats[0][0],stats[1][0]),(stats[0][1],stats[1][1]),phi,q0=q0)
            sel.append(int(np.argmax(cf.single_auction_profit(P))))
        except Exception: fails+=1
    out[str(f)]=dict(n_criteo=int(f*nC),n_ipinyou=int(f*nI),reps=len(sel),fails=fails,
                     freq={str(k):sel.count(k)/max(len(sel),1) for k in (0,1,2)},
                     mean_share=float(np.mean([share[k] for k in sel])) if sel else None)
    print(f,out[str(f)],flush=True)
json.dump(out,open('ref_learning_real.json','w'),indent=1)
