"""Clicks per impression by rival-bid bin (iPinYou 2013-03-16), bins matching the calibrated background law:
[0,75), [75,225), [225,300) -> grid points 0, h, 2h (bid 2h <-> 300)."""
import pandas as pd, numpy as np, json
rd=lambda f,cols,names: pd.read_csv(f,sep='\t',header=None,quoting=3,usecols=cols,dtype=str).set_axis(names,axis=1)
i=rd('imp.txt',[0,20],['id','pay']); c=rd('clk.txt',[0],['id'])
i['pay']=i.pay.astype(float); i['clk']=i.id.isin(set(c.id)).astype(int)
bins=[0,75,225,300.1]; i['bin']=pd.cut(i.pay,bins,right=False,labels=False)
g=i.groupby('bin').agg(n=('clk','size'),ctr=('clk','mean'),pay=('pay','mean'))
out={'ctr_by_bin':g.reset_index().to_dict('records')}
# bootstrap ratio of bins
rng=np.random.default_rng(0); rat=[]
arr=[i.clk[i.bin==b].values for b in range(3)]
for _ in range(300):
    m=[rng.choice(a,len(a)).mean() for a in arr]; rat.append([m[1]/m[0],m[2]/m[0]])
out['ratio_sd']=np.std(rat,0).tolist()
json.dump(out,open('ctr_bins.json','w'),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
