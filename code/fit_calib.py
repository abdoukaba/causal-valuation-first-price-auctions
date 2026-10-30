import json, numpy as np
from scipy.optimize import minimize
S=json.load(open('ipy_stats.json'))
pts=np.array([[float(k)/3000, v] for k,v in S['cdf_given_won'].items()])
F=lambda x:1/(1+np.exp(-x)); m=np.array([0,.05,.1])
def W(a,q,z): return sum(q[j]*F((a-m[j])/z) for j in range(3))
def unpack(t):
    q=np.exp(t[:3]); q/=q.sum(); return q, np.exp(t[3])
def loss(t):
    q,z=unpack(t); return np.sum((W(pts[:,0],q,z)/W(.1,q,z)-pts[:,1])**2)
best=min((minimize(loss,np.r_[np.random.default_rng(i).normal(size=3),np.log(z0)],method='Nelder-Mead',options=dict(maxiter=20000,xatol=1e-10,fatol=1e-14)) for i in range(20) for z0 in (.01,.03,.1,.2)),key=lambda r:r.fun)
q,z=unpack(best.x)
fit=W(pts[:,0],q,z)/W(.1,q,z)
print('q',q.round(4),'zeta',round(z,4),'rmse',np.sqrt(best.fun/len(pts)))
for (a,v),f in zip(pts,fit): print(round(a*3000),round(v,3),round(f,3))
# model's own ratio at Table-1 calibration
q1=np.array([.2,.4,.4]); print('model ratio h/2h (zeta .2, q0):',W(.05,q1,.2)/W(.1,q1,.2),' data:',S['cdf_given_won']['150'])
json.dump(dict(q=q.tolist(),zeta=z,rmse=float(np.sqrt(best.fun/len(pts))),fit=fit.tolist(),pts=pts.tolist()),open('fit_calib.json','w'),indent=1)
