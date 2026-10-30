"""Season 3, 2013-10-23 (advertisers 2821 and 2997; 2259 has no impression log for this date)."""
import pandas as pd, numpy as np, json
d=pd.read_pickle('d.pkl'); d=d[d.adv.isin(['2821','2997'])].copy(); d['ex']=d.ex.fillna('none'); d['vis']=d.vis.fillna('none'); d['fmt']=d.fmt.fillna('none'); d['ex']=d.ex.fillna('none')
c=pd.read_csv('clk.txt',sep='\t',header=None,quoting=3,usecols=[0],dtype=str)[0]
d['clk']=d.id.isin(set(c)).astype(int)
ts=pd.read_csv('bid.txt',sep='\t',header=None,quoting=3,usecols=[0,1],dtype=str).set_axis(['id','ts'],axis=1)
d=d.merge(ts,on='id'); d['hr']=d.ts.str[8:10]; d=d[d.hr!='21']
out={}
out['by_adv_ex']=[dict(adv=a,ex=e,bid=float(g.bid.iat[0]),n=len(g),win=float(g.won.mean()),
     pay_med=float(g.pay.median()) if g.won.sum() else None, floor_med=float(g.floor.median()))
     for (a,e),g in d.groupby(['adv','ex'])]
w=d[d.won==1].copy()
w['cell']=w.adv+'_'+w.ex+'_'+w.vis+'_'+w.fmt+'_'+w.hr
x=w.pay-w.groupby('cell').pay.transform('mean'); y=w.clk-w.groupby('cell').clk.transform('mean')
b=(x*y).sum()/(x*x).sum(); e=y-b*x; se=np.sqrt((e**2*x**2).sum())/(x*x).sum()
out['ctr_slope_per1000_per100']=dict(b=float(b*1e5),se=float(se*1e5),mean=float(w.clk.mean()*1000),n_wins=len(w),clicks=int(w.clk.sum()))
for a,g in w.groupby('adv'):
    x=g.pay-g.groupby('cell').pay.transform('mean'); y=g.clk-g.groupby('cell').clk.transform('mean')
    b=(x*y).sum()/(x*x).sum(); e=y-b*x; se=np.sqrt((e**2*x**2).sum())/(x*x).sum()
    out['slope_'+a]=dict(b=float(b*1e5),se=float(se*1e5),mean=float(g.clk.mean()*1000),clicks=int(g.clk.sum()))
# naive bid-level comparison for 2821
g=d[d.adv=='2821']; out['naive_2821']={str(int(k)):float(v) for k,v in g.groupby('bid').won.mean().items()}
json.dump(out,open('s3_summary.json','w'),indent=1)
print(json.dumps(out,indent=1))
