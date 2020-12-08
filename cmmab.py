import numpy as np
import scipy.stats as sp
import random


class CMMAB:
  """Implements several algorithms for the contextual multi-armed bandit problem.
  
  This class contains functions required for creating an instance of contextual
  multi-armed bandit problem and implements various well-known algorithms 
  in this setting.
  """	
  def __init__(self, T, k, d, sigma=1, means=None, context=None, labels=None):
    """
    Args:
      T: Horizon.     
      k: Number of arms.
      d: Dimension.
      means: A d by k matrix of arm parameters, each column representing one arm.
      sigma: The standard deviation of noise.
      context: A T by k by d matrix of contexts for all arms, at all time-periods.
      labels: Vector of correct labels for all contexts. This is only used when the
        contextual bandit problem is created using a classification dataset.
    """
    self.T = T
    self.k = k
    self.d = d
    self.means = means
    self.sigma = sigma
    self.context = context
    self.labels = labels
    self._gen_cont = 0
    if context is None: ## Generate Context
      self._gen_cont = 1
      self.context = np.repeat(np.random.randn(T, 1, d), k, axis=1)/np.sqrt(d)
    elif context.ndim == 2: ## Shared context
      self.context = np.zeros((T,k,d))
      for j in range(k):
        self.context[:,j,:] = context		
    if labels is not None: ## We have labels, so 0-1 reward
      self.obs = np.zeros((T,k))
      for t in range(T):
        self.obs[t, labels[t]] = 1
    else: ## Generate rewards
      self.obs = np.dot(self.context, means).diagonal(axis1=2) + \
          self.sigma * np.random.randn(T,k)
    return
    

  def rank_one_update(self, cov_mat_inv, LS_sol, x, y):
    """Implements Sherman-Morrison rank-one update for least squares estimations.
  
    This function implements the Sherman-Morrison rank-one formula to be used for quick
    least squares updates. 
    Args:
      cov_mat_inv: The inverse covariance matrix, which needs to be updated using
        Sherman-Morrison. 
      LS_sol: Previous solution of least squares.
      x: The new observed context.
      y: The new reward.
    Returns:
      cov_mat_inv_upd: The updated inverse covariance matrix.
      LS_sol_upd: The updated least-squares solution.
    """
    inn_prod = np.inner(x, np.dot(cov_mat_inv, x))
    out_prod = np.dot(cov_mat_inv, np.outer(x, x))
    cov_mat_inv_upd = cov_mat_inv - np.dot(out_prod, cov_mat_inv)/(1.0 + inn_prod)
    LS_sol_upd = np.dot(np.eye(self.d) - out_prod/(1.0 + inn_prod), LS_sol) + y * np.dot(
        cov_mat_inv_upd, x)
    return cov_mat_inv_upd, LS_sol_upd


  def greedy(self, sub_arm = None):
    """Implements the Greedy algorithm for contextual multi-armed bandit.

    This function implements the Greedy algorithm for contextual multi-armed bandit.
    The algorithm has a parameter that indicates how many arms (selected at random) 
    are pulled, allowing the implementation of subsampling. Each selected arm is pulled for
    d time-periods, where d is the dimension. After this, if the covariance matrix is full
    rank the least-squares estimation is used and will be updated using Sherman-Morrison
    formula. Otherwise, a ridge regression with penalty 0.01 is used.

    Args:
      sub_arm: Number of arms that are used. If None, it means all arms are used.

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """
    if sub_arm is None:
      sub_arm = self.k
    else:
      sub_arm = int(min(sub_arm,self.k))
    ind = random.sample(list(range(self.k)), sub_arm)
    est_mean = np.zeros((self.d, self.k))
    pulls = np.zeros(self.T)
    regret = np.zeros(self.T)
    num_pulls = np.zeros(self.k)
    cov_mat = np.zeros((self.d, self.d, self.k))
    cov_mat_inv = np.zeros((self.d, self.d, self.k))
    out_prod = np.zeros((self.d, self.k))
    for t in range(self.T):
      ## Step 1: Select an arm to pull
      if t <= self.d*sub_arm-1:
        a = ind[t%sub_arm]
      else:
        a = ind[np.dot(self.context[t,ind,:], est_mean[:,ind]).diagonal().argmax()]
      ## Step 2: Calculate Regret
      num_pulls[a] += 1
      if self.labels is None:
        best_arm = np.dot(self.context[t,:,:], self.means).diagonal().argmax()
        regret[t] = np.inner(self.context[t, best_arm, :], self.means[:, best_arm]) - np.inner(
        self.context[t, a, :], self.means[:, a])
      else:
        regret[t] = 1 - self.obs[t, a]
      pulls[t] = a
      cov_mat[:, :, a] += np.outer(self.context[t, a, :], self.context[t, a, :])
      out_prod[:, a] += self.obs[t,a] * self.context[t,a,:]
      ## Step 3: Update arm parameters            
      if t == self.d * sub_arm - 1:
        for j in range(sub_arm):
          arm = ind[j]
          try: ## If covariance matrix is invertible use least squares.
            est_mean[:,arm] = np.linalg.solve(cov_mat[:,:,arm], out_prod[:,arm])
            cov_mat_inv[:, :, arm] = np.linalg.inv(cov_mat[:, :, arm])
          except: ## If not, add a small 0.01 penalty and use ridge regression.
            est_mean[:,arm] = np.linalg.solve(
                cov_mat[:,:,arm] + 0.01 * np.eye(self.d), out_prod[:,arm])
            cov_mat_inv[:, :, arm] = np.linalg.inv(
                cov_mat[:, :, arm] + 0.01 * np.eye(self.d))
      elif t>= self.d * sub_arm:
        cov_mat_inv[:, :, a], est_mean[:, a] = self.rank_one_update(
            cov_mat_inv=cov_mat_inv[:, :, a], 
            LS_sol=est_mean[:, a], 
            x=self.context[t, a, :], 
            y=self.obs[t, a])

    return regret, pulls, num_pulls
				

  def ts(self, sub_arm=None, prior_scale=None):
    """Implements the Thompson Sampling algorithm for contextual multi-armed bandit.

    This implements the Thompson Sampling algorithm for contextual multi-armed bandit using
    gaussian priors/updates. 
    The algorithm has a parameter that indicates how many arms (selected at random) 
    are pulled, allowing the implementation of subsampling. The algorithm uses zero-mean
    gaussian priors for all the arms. The covariance matrix can be scaled using the
    second argument. Other than covariance matrices, the algorithm keeps track of their
    Cholesky factorizations.
    Args:
      sub_arm: Number of arms that are used. If None, it means all arms are used.
      prior_scale: The scaling factor of the gaussian priors.

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """
    if sub_arm is None:
      sub_arm = self.k
    else:
      sub_arm = int(min(sub_arm,self.k))
    ind = random.sample(list(range(self.k)), sub_arm)
    est_mean = np.zeros((self.d, sub_arm))
    pulls = np.zeros(self.T)
    regret = np.zeros(self.T)
    num_pulls = np.zeros(self.k)
    if prior_scale is None:
      if self._gen_cont == 1: ## In this case, we normalize the covariance matrices by d.
        nor_cov = np.repeat(
            np.eye(self.d).reshape(self.d, self.d, 1), sub_arm, axis = 2)/self.d
        nor_chols = np.repeat(
            np.eye(self.d).reshape(self.d, self.d, 1), sub_arm, axis = 2)/np.sqrt(self.d)
      else:
        nor_cov = np.repeat(
            np.eye(self.d).reshape(self.d, self.d, 1), sub_arm, axis = 2)
        nor_chols = np.repeat(
            np.eye(self.d).reshape(self.d, self.d, 1), sub_arm, axis = 2)
    else:
      nor_cov = np.repeat(
          np.eye(self.d).reshape(self.d, self.d, 1), sub_arm, axis = 2) * prior_scale**2
      nor_chols = np.repeat(
          np.eye(self.d).reshape(self.d, self.d, 1), sub_arm, axis = 2) * prior_scale
    nor_means = np.zeros((self.d, sub_arm))
    samples = np.zeros((self.d, sub_arm))
  
    ## Draw all the gaussian samples in the beginning to speed up the process.
    all_samples = np.random.randn(self.d, self.k, self.T)

    for t in range(self.T):
      ## Step 1: Draw samples
      for i in range(sub_arm):
        samples[:, i] = np.dot(nor_chols[:, :, i], all_samples[:, i, t]) + nor_means[:, i]
      ## Step 2: Select an arm to pull
      a = np.dot(self.context[t, ind, :], samples).diagonal().argmax()
      arm = ind[a]
      num_pulls[arm] += 1
      ## Step 3: Calculate regret
      if self.labels is None:
        best_arm = np.dot(self.context[t, :,:], self.means).diagonal().argmax()
        regret[t] = np.inner(
            self.context[t, best_arm, :], self.means[:, best_arm]) - np.inner(
            self.context[t, arm, :], self.means[:, arm])
      else:
        regret[t] = 1 - self.obs[t, arm]
      pulls[t] = arm
      ## Step 4: Update posteriors
      post_cov = np.linalg.pinv(np.linalg.pinv(nor_cov[:, :, a]) + np.outer(
          self.context[t, arm, :], self.context[t, arm, :]) / self.sigma**2)
      nor_means[:, a] = np.dot(post_cov, np.dot(np.linalg.pinv(
          nor_cov[:, :, a]), nor_means[:, a]) + \
          self.obs[t, arm] * self.context[t, arm, :] / self.sigma**2)
      nor_cov[:, :, a] = post_cov
    
      nor_chols[:, :, a] = np.linalg.cholesky(nor_cov[:, :, a])
    return regret, pulls, num_pulls
		

  def oful(self, sub_arm=None, lam=None):
    """Implements the OFUL algorithm for contextual multi-armed bandit.

    This implements the OFUL algorithm for contextual multi-armed bandit. For more 
    details see the original paper: 
      https://papers.nips.cc/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf
    Our implementation uses shrunk confidence sets that are feasible because of the
    special structure of our problem (especially of size d*log(k) rather than d*k).
    The algorithm has a parameter that indicates how many arms (selected at random) 
    are pulled, allowing the implementation of subsampling.
    Args:
      sub_arm: Number of arms that are used. If None, it means all arms are used.
      lam: The regularization parameter lambda for ridge regressions.
    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """

    if sub_arm is None:
      sub_arm = self.k
    else:
      sub_arm = int(min(sub_arm,self.k))

    if lam is None:
      lam = 0.1
    ind = random.sample(list(range(self.k)), sub_arm) # Random order
    est_mean = np.zeros((self.d, self.k))
    pulls = np.zeros(self.T)
    regret = np.zeros(self.T)
    num_pulls = np.zeros(self.k)
    cov_mat_inv = np.repeat(
      np.eye(self.d).reshape(self.d, self.d, 1), self.k, axis = 2) / lam
    out_prod = np.zeros((self.d, self.k))
    dim = self.d
      
    for t in range(self.T):
      ## Step 1: Calculate optimistic rewards
      rad = self.sigma * np.sqrt(self.d * np.log(2 * self.k * (t+1)**3)) + np.sqrt(lam)
      temp_rew = np.zeros(self.k)
      for i in range(self.k):
        temp_rew[i] = np.inner(self.context[t, i, :], np.dot(cov_mat_inv[:, :, i],
              self.context[t, i, :]))
      optimistic_rewards = np.dot(self.context[t, ind, :], est_mean[:, ind]).diagonal() + \
          rad * np.sqrt(temp_rew[ind])
      ## Step 2: Select an arm to pull
      a = ind[optimistic_rewards.argmax()]
      num_pulls[a] += 1
      ## Step 3: Calculate regret
      if self.labels is None:
        best_arm = np.dot(self.context[t, :, :], self.means).diagonal().argmax()
        regret[t] = np.inner(
            self.context[t, best_arm, :], self.means[:, best_arm]) - np.inner(
            self.context[t, a, :], self.means[:, a])
      else:
          regret[t] = 1 - self.obs[t, a]
      pulls[t] = a
      ## Step 4: Update covariance matrices using Sherman-Morrison.
      cov_mat_inv[:, :, a], est_mean[:, a] = self.rank_one_update(
      cov_mat_inv[:, :, a], est_mean[:, a], self.context[t, a, :], self.obs[t, a])
    return regret, pulls, num_pulls