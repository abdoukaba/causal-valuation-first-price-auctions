"""Arb evaluation of the Theorem 1 bound for a data-calibrated primitive set:
background law q0 on mgrid (may extend above the top bid), causal values Delta[s][m], offsets d[s], score scale zeta,
and an optional fixed KL reference with smallest probability rho_min (log 3 -> log(1/rho_min))."""
import types
from flint import arb
import cert_interval as ci
A, amax, amin = ci.A, ci.amax, ci.amin
def make_primitives(mgrid, q0, Delta, d, zeta):
    MG=[A(x) for x in mgrid]; Q0=[A(x) for x in q0]; K=len(MG); Z=A(zeta); DO=[A(x) for x in d]
    DL=[[A(x) for x in row] for row in Delta]
    def primitives(eta):
        eta=A(eta) if not isinstance(eta,arb) else eta
        w={(s,a,m):1/(1+(-(ci.ACTS[a]-MG[m]-DO[s])/Z).exp()) for s in range(2) for a in range(2) for m in range(K)}
        g={(s,a,m):w[s,a,m]*(DL[s][m]-ci.ACTS[a]) for s in range(2) for a in range(2) for m in range(K)}
        verts=[[(1-eta)*Q0[j]+(eta if j==k else 0) for j in range(K)] for k in range(3)]
        pc={}; LPsi=A(1); rplus=A(0)
        for a in range(2):
            ell=[[sum((v[m]*w[s,a,m] for m in range(K)),A(0)) for v in verts] for s in range(2)]
            lmin=amin([x for r in ell for x in r]); lmax=amax([x for r in ell for x in r])
            lo=[amin(ell[s]) for s in range(2)]; hi=[amax(ell[s]) for s in range(2)]
            gamma=amax([amax([g[s,a,m] for m in range(K)])-amin([g[s,a,m] for m in range(K)]) for s in range(2)])
            omega=amax([amax([w[s,a,m] for m in range(K)])-amin([w[s,a,m] for m in range(K)]) for s in range(2)])
            O1s,O0s=[],[]
            for i in range(K):
                for j in range(K):
                    d1=w[1,a,i]-w[1,a,j]; d0=w[0,a,i]-w[0,a,j]
                    for l0 in (lo[0],hi[0]):
                        for l1 in (lo[1],hi[1]):
                            O1s.append(abs(l0*d1-l1*d0)); O0s.append(abs((1-l0)*d1-(1-l1)*d0))
            beta=amax(O1s)/(4*lmin)+amax(O0s)/(4*(1-lmax))
            Lg=amax([abs(g[1,a,m]-g[0,a,m]) for m in range(K)]); Lw=amax([abs(w[1,a,m]-w[0,a,m]) for m in range(K)])
            for m in range(K):
                LPsi=amax([LPsi,w[1,a,m]/w[0,a,m],w[0,a,m]/w[1,a,m],(1-w[1,a,m])/(1-w[0,a,m]),(1-w[0,a,m])/(1-w[1,a,m])])
            rplus=amax([rplus]+[g[s,a,m] for s in range(2) for m in range(K)])
            if not (bool(lmin>A('0.1465')) and bool(lmax<A('0.8535'))):
                raise ValueError(f'Assumption 2 fails: lmin={float(lmin.mid()):.4f}, lmax={float(lmax.mid()):.4f}')
            pc[a]=dict(gamma=gamma,omega=omega,beta=beta,Lg=Lg,Lw=Lw,lmin=lmin,lmax=lmax)
        return pc,LPsi,rplus,None
    return primitives
def kappa(mgrid,q0,Delta,d,zeta,T,lam,eps,eta,theta,rho_min=None,nt=2000):
    prim=make_primitives(mgrid,q0,Delta,d,zeta)
    g=dict(ci.sharp_kappa.__globals__); g['primitives']=prim
    if rho_min is not None:
        logr=(1/A(rho_min)).log()
        class FA:
            def __init__(s,x): s.x=x
            def log(s): return logr
        g['arb']=lambda x: FA(x) if x==3 else arb(x)
    f=types.FunctionType(ci.sharp_kappa.__code__,g,None,ci.sharp_kappa.__defaults__)
    return f(T,lam,eps,eta,theta,nt)
def best(mgrid,q0,Delta,d,zeta,T,lam,eps,eta,thetas=('1','4','8','16','24','32'),rho_min=None,nt=2000):
    b=None
    for th in thetas:
        r=kappa(mgrid,q0,Delta,d,zeta,T,lam,eps,eta,th,rho_min,nt)
        k=float(r['kappa'].upper())
        if b is None or k<b[0]: b=(k,th,bool(r['certified']))
    return b
if __name__=='__main__':
    # regression: Table-1 primitives reproduce the paper's 0.9176 at eta=0.45 (2000 cells)
    D=[[0.06+0.03*m for m in (0,.05,.1)],[0.095+0.03*m for m in (0,.05,.1)]]
    print(best([0,.05,.1],[.2,.4,.4],[[str(x) for x in r] for r in D],['-0.01','0.01'],'0.2',1,'2.25','0.05','0.45',thetas=('16',)))
