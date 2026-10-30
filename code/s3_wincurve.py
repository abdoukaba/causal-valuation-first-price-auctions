"""Season 3, 2013-10-23, advertiser 2821: two bid levels (277, 294). Compare the win-rate ratio measured from
binary win/loss outcomes with the ratio implied by observed paying prices (second price: pay = max(M, floor))."""
import pandas as pd, numpy as np, json
d=pd.read_pickle('d.pkl'); d=d[d.adv=='2821'].copy()
ts=pd.read_csv('bid.txt',sep='\t',header=None,quoting=3,usecols=[0,1],dtype=str).set_axis(['id','ts'],axis=1)
d=d.merge(ts,on='id'); d['hr']=d.ts.str[8:10]
d=d[d.hr!='21']                     # no impressions logged in hour 21 for this advertiser
d['cell']=d.ex+'_'+d.vis+'_'+d.fmt+'_'+d.slot.where(d.slot.map(d.slot.value_counts())>500,'other')+'_'+d.hr
out={'n':len(d),'share_277':float((d.bid==277).mean())}
out['raw']={str(int(b)):float(g.won.mean()) for b,g in d.groupby('bid')}
# stratified (within-cell) comparison, cells with both bids
g=d.groupby(['cell','bid']).won.agg(['mean','size']).unstack()
g=g.dropna(); w=g[('size',277.0)]
r277=np.average(g[('mean',277.0)],weights=w); r294=np.average(g[('mean',294.0)],weights=w)
out['within_cell']=dict(win_277=float(r277),win_294=float(r294),ratio=float(r277/r294),cells=int(len(g)),n_277_covered=float(w.sum()/(d.bid==277).sum()))
# bootstrap over cells
rng=np.random.default_rng(0); rs=[]
for _ in range(400):
    s=g.sample(len(g),replace=True,random_state=int(rng.integers(1e9))); ww=s[('size',277.0)]
    rs.append(np.average(s[('mean',277.0)],weights=ww)/np.average(s[('mean',294.0)],weights=ww))
out['within_cell']['ratio_se']=float(np.std(rs))
# paying-price implied ratio among wins at 294, same cell weights
wn=d[(d.won==1)&(d.bid==294)]
fl=(wn.floor>=277).mean(); out['floor_ge_277_share_wins294']=float(fl)
pc=wn.groupby('cell').pay.apply(lambda s:(s<277).mean())
pc=pc.reindex(g.index).dropna(); ww=w.reindex(pc.index)
out['price_implied_ratio']=float(np.average(pc,weights=ww))
out['price_implied_ratio_raw']=float((wn.pay<277).mean())
out['pay_q_wins294']=np.percentile(wn.pay,[10,25,50,75,90]).tolist()
# local elasticity of win prob wrt bid
out['elasticity_dlogW_dlogb']=float(np.log(r294/r277)/np.log(294/277))
# assignment check: is bid level predictable from context? share of 277 by exchange/vis
out['share277_by_ex']=d.groupby('ex').bid.apply(lambda s:float((s==277).mean())).to_dict()
out['share277_by_hour']=d.groupby('hr').bid.apply(lambda s:round(float((s==277).mean()),3)).to_dict()
json.dump(out,open('s3_wincurve.json','w'),indent=1)
print(json.dumps(out,indent=1))
