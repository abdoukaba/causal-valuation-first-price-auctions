import numpy as np, grid_certificate as gc, monotone_test as mt
for eps in (0.05,0.03, 0.02, 0.01):
    m = gc.Model(21, 60, .5, eps, 1., 2.25); rng = np.random.default_rng(1)
    worst=[]
    for k in range(6):
        q = rng.dirichlet([1,1,1], size=m.nt); qt = mt.shift_down(rng, q, .3)
        d = mt.upper_tails(mt.phi(m,q)) - mt.upper_tails(mt.phi(m,qt))
        worst.append((d.min(), d.max(), np.argmin(d.min(1)), d[:, 0].min(), d[:, 1].min()))
    print(eps, [(f'{a:.2e}', f'{b:.2e}', i, f'{c:.1e}', f'{e:.1e}') for a,b,i,c,e in worst])
