"""Anchored logit equilibria in the data-calibrated market (Criteo offsets) on two grids."""
import json, numpy as np, calib_full as cf
P=cf.primitives()
refs={'logit (Shannon)':None,'uniform KL':np.array([1,1,1])/3,'0.6 on h':np.array([.2,.6,.2]),'0.9 on h':np.array([.05,.9,.05]),
      '0.96 on h':np.array([.02,.96,.02]),'0.9 on 0':np.array([.9,.05,.05]),'0.9 on 2h':np.array([.05,.05,.9])}
out={}
for grid in ((21,60),(31,80)):
    for name,rho in refs.items():
        qs=[]
        for sd in range(4):
            m=cf.cm.CalibModel(grid[0],grid[1],.45,.05,1.,2.25,cf.MG,P['q0'],P['zeta'],P['d'],P['Delta'],P['p0'],rho)
            s=m.solve(sd,max_iter=2000,damping=.4); qs.append(s['q'])
        eq=m.monetary(s['q'],s['pol']); br=m.monetary(s['q'])
        rmin=None if rho is None else float(rho.min())
        out[f'{grid[0]}x{grid[1]}|{name}']=dict(eq=eq,br=br,ratio=eq/br,spread=float(np.abs(np.array(qs)-qs[0]).max()),res=s['res'],
              rho_min=rmin,gap_bound=None if rmin is None else .05*2.25*np.log(1/rmin),gap=br-eq)
        print(grid,name,{k:(round(v,5) if isinstance(v,float) else v) for k,v in out[f'{grid[0]}x{grid[1]}|{name}'].items()},flush=True)
json.dump(out,open('calib_anchored_grid.json','w'),indent=1,default=float)
