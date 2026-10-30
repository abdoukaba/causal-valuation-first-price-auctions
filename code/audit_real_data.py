"""Audit supplied real-record resampling results and optionally repeat model calculations.

No observations, treatment assignments, clicks, or outcomes are generated.
The original record-level inputs are not distributed with this package.
Run from any directory; outputs go to ../results unless --output-dir is set.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    rad = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0., mid-rad), min(1., mid+rad)]

def summarize(records):
    rows = []
    for fraction, r in records.items():
        n = r['reps'] + r['fails']
        counts = {str(k): round(r['freq'][str(k)] * r['reps']) for k in range(3)}
        assert sum(counts.values()) == r['reps']
        for k in counts:
            assert abs(counts[k]/r['reps'] - r['freq'][k]) < 1e-10
        rows.append(dict(fraction=float(fraction), n_criteo=r['n_criteo'],
                         n_ipinyou=r['n_ipinyou'], attempts=n, estimable=r['reps'],
                         failures=r['fails'], counts=counts,
                         p_h_conditional=counts['1']/r['reps'],
                         p_h_all_attempts=counts['1']/n,
                         conditional_wilson95=wilson(counts['1'],r['reps']),
                         all_attempts_wilson95=wilson(counts['1'],n),
                         conditional_model_ratio=r['mean_share']))
    return rows

def write_figure(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.array([r['n_ipinyou'] for r in rows])
    unconditional=np.array([r['p_h_all_attempts'] for r in rows])
    intervals=np.array([r['all_attempts_wilson95'] for r in rows])
    fig, ax=plt.subplots(figsize=(7.0,3.8),layout='constrained')
    ax.errorbar(x,100*unconditional,
                yerr=100*np.vstack((unconditional-intervals[:,0],intervals[:,1]-unconditional)),
                fmt='o-',color='#205680',capsize=3,label='Select h / all 100 draws')
    ax.plot(x,100*np.array([r['p_h_conditional'] for r in rows]),'s--',
            color='#ba6324',label='Select h / estimable draws')
    ax.plot(x,[r['estimable'] for r in rows],'^:',color='#555555',label='Estimable / all 100 draws')
    ax.set_xscale('log'); ax.set_ylim(-2,104)
    ax.set_xlabel('iPinYou auctions resampled per draw (log scale)')
    ax.set_ylabel('Percentage of draws')
    ax.set_xticks(x,[f'{v:,}' for v in x],rotation=25,ha='right')
    ax.grid(axis='y',alpha=.2); ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    fig.savefig(destination)
    plt.close(fig)

def verify_models(destination):
    from calib_model import CalibModel
    p=json.loads((ROOT/'results/calib_full.json').read_text())['criteo_offsets']['primitives']
    stored=json.loads((ROOT/'results/calib_anchored_grid.json').read_text())
    references={'logit (Shannon)':None, 'uniform KL':np.ones(3)/3,
                '0.6 on h':[.2,.6,.2], '0.9 on h':[.05,.9,.05],
                '0.96 on h':[.02,.96,.02], '0.9 on 0':[.9,.05,.05],
                '0.9 on 2h':[.05,.05,.9]}
    outputs={}
    for nb, nt in [(21,60),(31,80)]:
        for name,rho in references.items():
            m=CalibModel(nb,nt,.45,.05,1.,2.25,[0.,.05,.1,.15],
                         p['q0'],p['zeta'],p['d'],p['Delta'],p['p0'],rho)
            solution=m.solve(0,max_iter=2000,damping=.4)
            eq=m.monetary(solution['q'],solution['pol']); br=m.monetary(solution['q'])
            key=f'{nb}x{nt}|{name}'
            result=dict(eq=eq,br=br,ratio=eq/br,gap=br-eq,residual=solution['res'])
            result['max_difference_from_supplied']=max(abs(result[k]-stored[key][k]) for k in ['eq','br','ratio','gap'])
            assert result['max_difference_from_supplied']<1e-7
            assert result['residual']<1e-8
            outputs[key]=result
            print(key,result,flush=True)
    (destination/'calibrated_model_recheck.json').write_text(json.dumps(outputs,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--verify-model',action='store_true')
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=summarize(json.loads((ROOT/'results/ref_learning_real.json').read_text()))
    result={'provenance':'Derived from supplied real-record resampling summaries; raw observations not refitted.',
            'interval_scope':'95% Wilson intervals describe Monte Carlo draw frequencies conditional on the supplied datasets and fixed fitted nuisance functions; not population generalization intervals.',
            'rows':rows}
    (args.output_dir/'real_reference_audit.json').write_text(json.dumps(result,indent=2))
    for r in rows:
        print(r['fraction'],r['estimable'],r['counts'],r['p_h_all_attempts'],flush=True)
    if args.figure:
        args.figure.parent.mkdir(parents=True,exist_ok=True);write_figure(rows,args.figure)
    if args.verify_model:
        verify_models(args.output_dir)

if __name__=='__main__':
    main()
