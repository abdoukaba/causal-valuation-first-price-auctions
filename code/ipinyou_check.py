import numpy as np, pandas as pd, json
d=pd.read_csv('slim.tsv',sep='\t',header=None,names=['ex','vis','fmt','floor','bid','pay','adv','clk','conv'],
              dtype={'adv':str})
d['clk']=(d.clk>0).astype(int); d['conv']=(d.conv>0).astype(int)
out={'n':len(d)}
amap={a:i for i,a in enumerate(d.adv.value_counts().index)}
d['A']=d.adv.map(amap)
out['by_adv']={}
for a,g in d.groupby('A'):
    p=g.pay.values
    out['by_adv'][int(a)]=dict(n=len(g),ctr=g.clk.mean(),cvr=g.conv.mean(),
        pay_q=dict(zip(['p10','p25','p50','p75','p90','p99'],np.percentile(p,[10,25,50,75,90,99]).tolist())),
        mean_pay=p.mean(), floor_binds=float((g.pay<=g.floor).mean()), zero_floor=float((g.floor==0).mean()))
p=d.pay.values
out['pay_q']=dict(zip(['p5','p10','p25','p50','p75','p90','p95','p99','max'],np.percentile(p,[5,10,25,50,75,90,95,99,100]).tolist()))
out['floor_binds']=float((d.pay<=d.floor).mean())
# truncated CDF: P(M<a | M<300)
grid=[25,50,75,100,150,200,250,300]
out['cdf_given_won']={a:float((p<a).mean()) for a in grid}
# CTR by pay decile, overall and within adv x exchange
d['dec']=pd.qcut(d.pay.rank(method='first'),10,labels=False)
out['ctr_by_pay_decile']=d.groupby('dec').agg(pay=('pay','median'),ctr=('clk','mean'),n=('clk','size')).reset_index().to_dict('records')
# within-cell slope: logistic-free linear regression of clk on pay with adv x exchange x slot-format fixed effects (demeaning)
d['cell']=d.A.astype(str)+'_'+d.ex.astype(str)+'_'+d.vis.astype(str)+'_'+d.fmt.astype(str)
x=d.pay-d.groupby('cell').pay.transform('mean'); y=d.clk-d.groupby('cell').clk.transform('mean')
b=(x*y).sum()/(x*x).sum(); e=y-b*x; se=np.sqrt((e**2*x**2).sum())/(x*x).sum()
out['within_slope_ctr_per_100']=dict(b=float(b*100),se=float(se*100),mean_ctr=float(d.clk.mean()))
for a,g in d.groupby('A'):
    x=g.pay-g.groupby('cell').pay.transform('mean'); y=g.clk-g.groupby('cell').clk.transform('mean')
    b=(x*y).sum()/(x*x).sum(); e=y-b*x; se=np.sqrt((e**2*x**2).sum())/(x*x).sum()
    out['by_adv'][int(a)]['within_slope_ctr_per_100']=dict(b=float(b*100),se=float(se*100))
    # ctr in low vs high pay half within cells
    hi=g.pay>g.groupby('cell').pay.transform('median')
    out['by_adv'][int(a)]['ctr_low_high']=[float(g.clk[~hi].mean()),float(g.clk[hi].mean())]
json.dump(out,open('ipy_stats.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float)[:6000])
