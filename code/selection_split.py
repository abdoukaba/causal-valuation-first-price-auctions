"""Table 9: exact two-channel decomposition of causal adverse selection (Proposition 4) at p=1/2, Q=q0."""
import numpy as np, json
F=lambda x:1/(1+np.exp(-x)); m=np.array([0,.05,.1]); Q=np.array([.2,.4,.4]); d=[-.01,.01]
D=[.06+.03*m,.095+.03*m]; p=[.5,.5]; out={}
for a in (.05,.1):
    w=[F((a-m-d[s])/.2) for s in (0,1)]; W=sum(p[s]*Q@w[s] for s in (0,1))
    within=sum(p[s]*(Q@(D[s]*w[s])-(Q@D[s])*(Q@w[s])) for s in (0,1))/W
    between=p[0]*p[1]*(Q@D[1]-Q@D[0])*(Q@w[1]-Q@w[0])/W
    ED=sum(p[s]*Q@D[s] for s in (0,1)); EDW=sum(p[s]*Q@(D[s]*w[s]) for s in (0,1))/W
    out[a]=dict(PrW=W,ED=ED,ED_win=EDW,gap=EDW-ED,within=within,between=between,between_share=between/(within+between))
    assert abs((within+between)-(EDW-ED))<1e-14
json.dump(out,open('../results/selection_split.json','w'),indent=1); print(json.dumps(out,indent=1))
