"""Numerical checks for the finite-action weighted Gibbs variational identity.
These are unit tests, not a PDE uniqueness certificate.
"""
import numpy as np
from scipy.special import logsumexp
rng = np.random.default_rng(20261028)
for k in (2,3,5):
 for _ in range(300):
  eps=float(rng.uniform(.02,1.))
  u=rng.normal(0,.2,k)
  rho=rng.dirichlet(np.ones(k))
  scaled=(u/eps)+np.log(rho)
  prob=np.exp(scaled-logsumexp(scaled))
  lognorm=eps*logsumexp(scaled)
  variational=float(prob@u-eps*np.sum(prob*np.log(prob/rho)))
  assert np.isclose(lognorm,variational,atol=1e-11)
  assert variational<=max(u)+1e-11
  assert max(u)-variational<=eps*np.log(1/min(rho))+1e-11
  monetary_gap=max(u)-float(prob@u)
  comparator_bound=eps*np.log(1/rho[int(np.argmax(u))])
  assert monetary_gap >= -1e-12
  assert monetary_gap <= comparator_bound + 1e-11
  dv=rng.normal(0,.001,k)
  pp=np.exp(scaled+dv/eps-logsumexp(scaled+dv/eps))
  assert sum(abs(pp-prob)) <= max(abs(dv))/eps + 1e-10
print('Weighted Gibbs identity, monetary gap, and softmax sensitivity: 900 randomized cases passed.')
