"""iPinYou season 1, 2013-03-16: join bid and impression logs on bid ID -> win/loss for every auction entered;
absolute rival-bid law, win curve, value vs competition."""
import pandas as pd, numpy as np, json
rd=lambda f,cols,names: pd.read_csv(f,sep='\t',header=None,quoting=3,usecols=cols,names=None,dtype=str).set_axis(names,axis=1)
b=rd('bid.txt',[0,7,14,15,16,17,18],['id','ex','vis','fmt','floor','cr','bid'])
i=rd('imp.txt',[0,20,21],['id','pay','adv'])
c=rd('clk.txt',[0],['id']); v=rd('conv.txt',[0],['id'])
for x in (b,): x['floor']=x.floor.astype(float); x['bid']=x.bid.astype(float)
i['pay']=i.pay.astype(float)
out=dict(n_bids=len(b),n_imps=len(i),n_clk=len(c),n_conv=len(v),bid_values=b.bid.value_counts().head(5).to_dict(),
         dup_bid_ids=int(b.id.duplicated().sum()),imp_not_in_bids=int((~i.id.isin(b.id)).sum()))
d=b.merge(i,on='id',how='left'); d['won']=d.pay.notna().astype(int)
d['clk']=d.id.isin(set(c.id)).astype(int); d['conv']=d.id.isin(set(v.id)).astype(int)
# advertiser for lost bids via creative -> advertiser map learned from wins
cmap=d[d.won==1].groupby('cr').adv.agg(lambda s:s.mode().iat[0]); d['adv']=d.cr.map(cmap)
out['adv_unmapped_share']=float(d.adv.isna().mean())
out['win_rate']=float(d.won.mean())
out['floor_above_bid']=float((d.floor>d.bid).mean())
out['win_rate_by_ex']=d.groupby('ex').won.mean().to_dict()
out['win_rate_by_adv']=d.groupby('adv').won.mean().to_dict()
# absolute CDF of M for a<300: P(M<a)=P(won)*P(pay<a | won)  (second price; pay=max(M,floor))
w=d[d.won==1]; wr=d.won.mean()
grid=[5,10,25,50,75,100,150,200,250,300]
out['abs_cdf']={a:float(wr*(w.pay<a).mean()) for a in grid}
out['pay_q_won']=dict(zip(['p10','p25','p50','p75','p90'],np.percentile(w.pay,[10,25,50,75,90]).tolist()))
out['floor_binds_won']=float((w.pay<=w.floor).mean())
# CTR vs pay, within cells
w=w.copy(); w['cell']=w.adv.astype(str)+'_'+w.ex+'_'+w.vis+'_'+w.fmt
x=w.pay-w.groupby('cell').pay.transform('mean'); y=w.clk-w.groupby('cell').clk.transform('mean')
bb=(x*y).sum()/(x*x).sum(); e=y-bb*x; se=np.sqrt((e**2*x**2).sum())/(x*x).sum()
out['ctr_within_slope_per1000_per100']=dict(b=float(bb*1e5),se=float(se*1e5),mean_ctr_per1000=float(w.clk.mean()*1000))
w['dec']=pd.qcut(w.pay.rank(method='first'),10,labels=False)
out['ctr_by_pay_decile']=w.groupby('dec').agg(pay=('pay','median'),ctr1000=('clk',lambda s:1000*s.mean())).round(3).to_dict('records')
# does losing depend on observables? (selection into loss)
out['win_rate_by_vis']=d.groupby('vis').won.mean().round(3).to_dict()
json.dump(out,open('season1_day.json','w'),indent=1,default=float)
np.save('pay_won.npy',d.loc[d.won==1,'pay'].values)
print(json.dumps(out,indent=1,default=float))

# Input for real-record reference resampling. Lost-auction clicks are unavailable;
# ref_learning_real.py uses click values only when won is true.
np.savez_compressed('ipy_s1.npz', won=d.won.to_numpy(dtype=bool),
                    pay=d.pay.to_numpy(dtype=float), clk=d.clk.to_numpy(dtype=np.int8))
