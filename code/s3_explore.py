import pandas as pd, numpy as np
rd=lambda f,cols,names: pd.read_csv(f,sep='\t',header=None,quoting=3,usecols=cols,dtype=str).set_axis(names,axis=1)
b=rd('bid.txt',[0,7,11,14,15,16,18,19],['id','ex','slot','vis','fmt','floor','bid','adv'])
i=rd('imp.txt',[0,20],['id','pay'])
b['bid']=b.bid.astype(float); b['floor']=b.floor.astype(float)
d=b.merge(i,on='id',how='left'); d['won']=d.pay.notna().astype(int); d['pay']=d.pay.astype(float)
print('imps not in bids',(~i.id.isin(b.id)).sum(),'dup',b.id.duplicated().sum())
print(d.groupby('adv').agg(n=('won','size'),win=('won','mean'),nbids=('bid','nunique'),bmin=('bid','min'),bmed=('bid','median'),bmax=('bid','max')))
for a,g in d.groupby('adv'):
    print(a, g.bid.value_counts().head(8).to_dict())
print('floor>bid',(d.floor>d.bid).mean(), 'win when floor>bid', d[d.floor>d.bid].won.mean())
d.to_pickle('d.pkl')
