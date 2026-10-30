"""Reproducible first-price Bayesian auction MFG: primitive certificate + PDE-PIDE grid.
This is a deliberately WEAKLY interacting, synthetic auction, not an estimate
of parameters from real exchange logs and not a numerical PDE error certificate.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, csv, math
import numpy as np
from scipy.special import expit, logsumexp
import fast_kernels as fk

ROOT=Path(__file__).resolve().parent

class Auction:
    def __init__(self, np_grid=41, nt=40, eta=.05, epsilon=.5):
        self.T=.2; self.lam=.1; self.entropy=epsilon; self.sigma=.2; self.eta=eta
        self.actions=np.array([0.,.05,.10]);self.nact=3
        self.budgets=np.round(np.arange(5)*.05,8); self.nb=5
        self.p=np.linspace(0,1,np_grid);self.np=np_grid;self.h=self.p[1]-self.p[0]
        self.nt=nt;self.dt=self.T/nt
        self.base=np.array([.2,.4,.4])
        self.delta=np.array([-.01,.01]);self.noise=.20
        self.D=.1;self.A=.1;self.R=self.D+self.A
        # Markov chain generator approximation of continuous Bayesian diffusion:
        # dP = sigma P(1-P) dI, no reflecting flux, natural degenerate endpoints.
        self.up=(self.sigma*self.p*(1-self.p))**2/(2*self.h**2)
        self.up[-1]=0.
        self.down=self.up.copy();self.down[0]=0.
        assert self.dt*(self.lam+max(self.up+self.down))<.99
        self.w=np.zeros((2,self.nact,self.nact))
        self.value=np.zeros((2,self.nact))
        for s in range(2):
            self.value[s]=(.060 if s==0 else .095)+.03*self.actions
            for ia,a in enumerate(self.actions):
                if ia>0:
                    self.w[s,ia]=expit((a-self.actions-self.delta[s])/self.noise)
        self.cert=self.certificate()

    def certificate(self):
        # Conservatively c<=w_s(a,m)<=1-c for a>0 and all s,m.
        c=min(float(self.w[:,1:,:].min()),float((1-self.w[:,1:,:]).min()))
        R=self.R;T=self.T;lam=self.lam;eps=self.entropy;J=self.nact;eta=self.eta
        # Certification use simpler conservative lower bound c0<=c.
        c0=.36
        assert c>=c0, (c,c0)
        M=lam*T*(R+eps*math.log(J))
        Lp=2*M
        CV=(R+M)*lam*T*eta
        CQ=eta*(R+M+Lp/c0)+CV
        CPiq=2*CQ/eps
        CQp=2*R+2*M+Lp/(c0*c0)
        CPix=max(2.,2*CQp/eps)
        CKx=2+1/(c0*c0)
        CKq=eta*(1+1/c0)
        alpha=self.sigma**2/2+lam*(CKx-1+CPix)
        Cm=lam*T*math.exp(alpha*T)*(CKq+CPiq)
        kappa=CPiq+CPix*Cm
        return dict(c_actual=c,c_cert=c0,R=R,M=M,Lp=Lp,CV=CV,CQ=CQ,Cpiq=CPiq,CQp=CQp,Cpix=CPix,CKx=CKx,CKq=CKq,alpha=alpha,Cm=Cm,kappa=kappa,certified=bool(kappa<1))

    def population_market(self, qt):
        mix=(1-self.eta)*self.base+self.eta*qt
        # w_s(a)=win probability conditioned on private binary latent S=s
        ws=np.einsum('sac,c->sa',self.w,mix)
        gs=np.einsum('sac,sc,c->sa',self.w,self.value,mix)-self.actions[None,:]*ws
        p=self.p[:,None]
        pw=(1-p)*ws[0][None,:]+p*ws[1][None,:]
        g=(1-p)*gs[0][None,:]+p*gs[1][None,:]
        denom=np.maximum(pw,1e-100)
        postW=np.clip(p*ws[1][None,:]/denom,0,1)
        denL=np.maximum(1-pw,1e-100)
        postL=np.clip(p*(1-ws[1][None,:])/denL,0,1)
        pw[:,0]=0;g[:,0]=0;postW[:,0]=self.p;postL[:,0]=self.p
        return pw,g,postW,postL

    def interp(self, v, bids, beliefs):
        bidx=np.maximum(0, bids.astype(int))
        jf=np.minimum(np.floor(beliefs/self.h).astype(int),self.np-2)
        frac=np.clip((beliefs-self.p[jf])/self.h,0,1)
        return (1-frac)*v[bidx,jf]+frac*v[bidx,jf+1]

    def generator(self,v):
        res=np.zeros_like(v)
        # row per budget, across belief
        res[:,:-1]+=self.up[:-1][None,:]*(v[:,1:]-v[:,:-1])
        res[:,1:]+=self.down[1:][None,:]*(v[:,:-1]-v[:,1:])
        return res

    def scores(self, v, qt):
        pw,g,pw_bel,pl_bel=self.population_market(qt)
        if fk.HAVE_NUMBA:
            return fk.scores_kernel(np.ascontiguousarray(v,dtype=float),pw,g,pw_bel,pl_bel,float(self.h),self.p,self.nact)
        score=np.full((self.nb,self.np,self.nact),-1e6)
        for ib in range(self.nb):
            for ia in range(min(ib,2)+1):
                if ia==0:
                    score[ib,:,ia]=v[ib]
                    continue
                vw=self.interp(v,np.full(self.np,ib-ia),pw_bel[:,ia])
                vl=self.interp(v,np.full(self.np,ib),pl_bel[:,ia])
                score[ib,:,ia]=g[:,ia]+pw[:,ia]*vw+(1-pw[:,ia])*vl
        return score

    def backward(self,qflow):
        V=np.zeros((self.nt+1,self.nb,self.np));pol=np.zeros((self.nt,self.nb,self.np,self.nact))
        for t in range(self.nt-1,-1,-1):
            score=self.scores(V[t+1],qflow[t])
            # subtract continuation, so a=0 has zero auction advantage
            adv=score-V[t+1,:,:,None]
            if fk.HAVE_NUMBA:
                pol[t],lse=fk.gibbs_kernel(adv/self.entropy);hj=self.entropy*lse
            else:
                probs=np.exp(adv/self.entropy-logsumexp(adv/self.entropy,axis=-1,keepdims=True))
                pol[t]=probs
                hj=self.entropy*logsumexp(adv/self.entropy,axis=-1)
            V[t]=V[t+1]+self.dt*(self.generator(V[t+1])+self.lam*hj)
        return V,pol

    def monetary_policy_value(self, qflow, policy=None):
        v=np.zeros((self.nt+1,self.nb,self.np))
        for t in range(self.nt-1,-1,-1):
            scores=self.scores(v[t+1],qflow[t])
            if policy is None:
                payoff=np.max(scores,axis=-1)-v[t+1]
            else:
                payoff=np.sum(policy[t]*np.where(scores < -1e5,0.0,scores-v[t+1,:,:,None]),axis=-1)
            v[t]=v[t+1]+self.dt*(self.generator(v[t+1])+self.lam*payoff)
        return float(v[0,-1,self.np//2])

    def forward(self,qflow,pol):
        mass=np.zeros((self.nt+1,self.nb,self.np))
        mass[0,-1,self.np//2]=1.
        qnew=np.zeros_like(qflow)
        for t in range(self.nt):
            m=mass[t];nextm=m.copy()
            qnew[t]=(m[:,:,None]*pol[t]).sum((0,1))
            up=self.dt*self.up[:-1][None,:]*m[:,:-1]
            dn=self.dt*self.down[1:][None,:]*m[:,1:]
            nextm[:,:-1]-=up;nextm[:,1:]+=up
            nextm[:,1:]-=dn;nextm[:,:-1]+=dn
            pw,_,bw,bl=self.population_market(qflow[t]);jump=self.lam*self.dt
            nextm-=jump*m
            if fk.HAVE_NUMBA:
                fk.jump_scatter(nextm,m,np.ascontiguousarray(pol[t]),pw,bw,bl,jump,float(self.h),self.p)
                mass[t+1]=nextm
                continue
            for ib in range(self.nb):
                for ia in range(min(ib,2)+1):
                    amounts=jump*m[ib]*pol[t,ib,:,ia]
                    if ia==0:
                        nextm[ib]+=amounts
                        continue
                    for win,prob,bels,newb in ((True,pw[:,ia],bw[:,ia],ib-ia),(False,1-pw[:,ia],bl[:,ia],ib)):
                        src=amounts*prob
                        jf=np.minimum(np.floor(bels/self.h).astype(int),self.np-2)
                        frac=np.clip((bels-self.p[jf])/self.h,0,1)
                        np.add.at(nextm[newb],jf,src*(1-frac));np.add.at(nextm[newb],jf+1,src*frac)
            mass[t+1]=nextm
        return mass,qnew

    def solve(self,seed=0,tol=1e-10,max_iter=100,damping=.7):
        rng=np.random.default_rng(seed);flow=rng.dirichlet([1,1,1],size=self.nt)
        residuals=[]
        for it in range(max_iter):
            V,pol=self.backward(flow);mass,qnew=self.forward(flow,pol)
            res=float(np.max(np.abs(qnew-flow)));residuals.append(res)
            if res<tol:break
            flow=(1-damping)*flow+damping*qnew
        V,pol=self.backward(flow);mass,qnew=self.forward(flow,pol)
        return dict(q=flow,V=V,pol=pol,m=mass,res=float(np.max(np.abs(qnew-flow))),steps=len(residuals),trace=residuals)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--np',type=int,default=41);parser.add_argument('--nt',type=int,default=40)
    parser.add_argument('--eta',type=float,default=.05);parser.add_argument('--epsilon',type=float,default=.5);parser.add_argument('--starts',type=int,default=8)
    args=parser.parse_args();model=Auction(args.np,args.nt,args.eta,args.epsilon)
    sols=[model.solve(i) for i in range(args.starts)]
    ref=sols[0];mass=ref['m']
    spread=max(float(abs(ref['q']-s['q']).max()) for s in sols)
    monetary=model.monetary_policy_value(ref['q'],ref['pol'])
    best_response=model.monetary_policy_value(ref['q'],None)
    out={"monetary_profit":monetary,"monetary_best_response":best_response,"monetary_exploitability":best_response-monetary,"model":"finite bid grid, continuous Bayesian belief diffusion, hard budget, one noisy-scored first-price rival", "n_belief_grid":args.np,"n_time_steps":args.nt,"modelled_rival_fraction":args.eta,"equilibrium_residual":ref['res'],"max_multistart_disagreement":spread,"max_mass_error":float(abs(mass.sum((1,2))-1).max()),"minimum_mass":float(mass.min()),"value_at_initial_state":float(ref['V'][0,-1,model.np//2]),"initial_population_bid_shares":ref['q'][0].tolist(),"terminal_population_bid_shares":ref['q'][-1].tolist(),"certificate":model.cert}
    tag=f'p{args.np}_t{args.nt}_eta{args.eta:g}_eps{args.epsilon:g}' if args.epsilon!=.5 else f'p{args.np}_t{args.nt}_eta{args.eta:g}'
    (ROOT/f'summary_{tag}.json').write_text(json.dumps(out,indent=2))
    with open(ROOT/f'bid_shares_{tag}.csv','w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time','bid_0','bid_0.05','bid_0.10']);writer.writerows(([j*model.dt,*row] for j,row in enumerate(ref['q'])))
    print(json.dumps(out,indent=2));return out
if __name__=='__main__':main()
