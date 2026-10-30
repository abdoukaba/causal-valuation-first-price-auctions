"""Calibrate the model's background rival law and score scale to iPinYou 2013-03-16 (bid 300 <-> 2h = 0.10),
then evaluate the Arb bound of Theorem 1."""
import sys, json, numpy as np; sys.path.insert(0,'.')
from scipy.optimize import minimize_scalar
import cert_general as cg
S=json.load(open('season1_day.json')); pay=np.load('pay_won.npy'); wr=S['win_rate']
cdf=lambda a: wr*(pay<a).mean()
out={'win_rate':wr}
F=lambda x:1/(1+np.exp(-x)); A_=np.arange(5,301,5)
for top in (450,600):
    mg=np.array([0,150,300,top])/3000
    q0=np.array([cdf(75), cdf(225)-cdf(75), wr-cdf(225), 1-wr])
    W=lambda a,z: sum(q0[j]*F((a-mg[j])/z) for j in range(4))
    emp=np.array([cdf(a) for a in A_])
    r=minimize_scalar(lambda lz: np.mean((W(A_/3000,np.exp(lz))-emp)**2),bounds=(np.log(.003),np.log(.5)),method='bounded')
    z=float(np.exp(r.x)); res=dict(q0=q0.round(4).tolist(),zeta_fit=z,rmse=float(np.sqrt(r.fun)),
        W_at_bids_model=[float(W(.05,z)),float(W(.1,z))],W_at_bids_data=[cdf(150),cdf(300)],cert={})
    for zeta in (round(z,4),0.1,0.2):
        for eta in ('0.3','0.45'):
            key=f'zeta={zeta},eta={eta}'
            try:
                best=None
                for th in ('4','8','16','32'):
                    k=cg.kappa(mg.round(6).tolist(),[str(round(x,6)) for x in q0[:-1]]+[str(round(1-sum(round(x,6) for x in q0[:-1]),6))],str(zeta),1,'2.25','0.05',eta,th)
                    if best is None or float(k['kappa'].upper())<best[0]: best=(float(k['kappa'].upper()),th,bool(k['certified']))
                res['cert'][key]=dict(kappa_upper=best[0],theta=best[1],verified=best[2])
            except ValueError as e: res['cert'][key]=str(e)
            print(top,key,res['cert'][key],flush=True)
    out[f'top={top}']=res; print(top,{k:v for k,v in res.items() if k!='cert'},flush=True)
json.dump(out,open('calib_season1.json','w'),indent=1)
