import numpy as np
import random

class MMAB:
  """Implements various algorithms for the stochastic multi-armed bandit problem.
  
  This class contains functions required for creating an instance of stochastic
  multi-armed bandit problem and implements various well-known algorithms 
  in this setting.
  """
  
  def __init__(self, T, k, means, sigma=1, binary=False):
    """Initializes an instance of stochastic multi-armed bandit problem.

    Args:
      T: Horizon.     
      k: Number of arms.
      means: A vector containing the mean reward of arms.
      sigma: The standard deviation of noise.
      binary: A boolean variable indicating if the reward is binary or gaussian.
    """
    self.T = T
    self.k = k
    self.means = means
    self.sigma = sigma
    self.binary = binary
    obs = np.zeros((T,k))
    for j in range(self.k):
      if(self.binary == 1):
        obs[:,j] = np.random.binomial(1, self.means[j], size=T)
        self.sigma = 0.5
      else:
        obs[:,j] = self.means[j] + self.sigma * np.random.randn(T)
    self.obs = obs
    return


  def greedy(self, sub_arm=None):
    """Implements the Greedy algorithm for a stochastic multi-armed bandit instance.

    This function implements the Greedy algorithm for stochastic multi-armed bandit.
    The algorithm has a parameter that indicates how many arms (selected at random) 
    are pulled, allowing the implementation of subsampling.

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
      sub_arm = int(min(sub_arm, self.k))

    ind = random.sample(list(range(self.k)), sub_arm)
    est_mean = np.zeros(self.k)
    pulls = np.zeros(self.T)
    regret = np.zeros(self.T)
    num_pulls = np.zeros(self.k)

    for t in range(self.T):
      if t <= sub_arm-1:
        a = ind[t]
        pulls[t] = a
        est_mean[a] = self.obs[t, a]
      else:
        a = est_mean.argmax()
        pulls[t] = a
        est_mean[a] = (self.obs[t, a] + est_mean[a] * num_pulls[a]) / (num_pulls[a] + 1)
      num_pulls[a] += 1
      regret[t] = max(self.means) - self.means[a]
    return regret, pulls, num_pulls


  def generic_ucb(self, explore, explore_log, sub_arm=None):
    """Implements a generic UCB algorithm.

    This function implements the upper confidence bound (UCB) algorithm for 
    stochastic multi-armed bandit. The algorithm has three parameters. 
    The first two vectors of parameters indicate the amount of exploration and 
    the last term shows whether subsampling is activated or not. 
    The exploration term for arm i at time t, in this generic implementation is:
       standard deviation * sqrt(explore[t] * log(explore_log[t]) / #pulls of arm i).
    For example, the well-known UCB1 algorithm corresponds go explore[t] = 2 and
    explore_log[t] = t.

    Args:
      explore: The vector of exploration terms that appears inside the square root 
        in the definition of the upper confidence bound algorithm.
      explore_log: The vector of exploration terms that appears inside the logarithm
        within the square root in the definition of the upper confidence bound 
        algorithm.
      sub_arm: Number of arms that are used. If None, it means all arms are used.

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """
    if sub_arm is None:
      sub_arm = self.k
    else:
      sub_arm = int(min(sub_arm, self.k))

    ind = random.sample(list(range(self.k)), sub_arm)   
    est_mean = np.zeros(self.k)
    pulls = np.zeros(self.T)
    regret = np.zeros(self.T)
    num_pulls = np.zeros(self.k)
    
    for t in range(self.T):
      if t <= sub_arm-1:
        a = ind[t]
        pulls[t] = a
        est_mean[a] = self.obs[t, a]
      else:
        optimistic_rewards = est_mean[ind] + self.sigma * np.sqrt(
            explore[t] * np.log(explore_log[t]) / num_pulls[ind])
        ind_max = optimistic_rewards.argmax()
        a = ind[ind_max]
        pulls[t] = a
        est_mean[a] = (self.obs[t, a] + est_mean[a] * num_pulls[a]) / (num_pulls[a] + 1)
      num_pulls[a] += 1
      regret[t] = max(self.means) - self.means[a]
    return regret, pulls, num_pulls

  def generic_ucb_v(self, explore, sub_arm=None):
    """Implements a generic UCB-V algorithm.

    This function implements the upper confidence bound (UCB-V) algorithm for 
    stochastic multi-armed bandit which takes into the account the variance of reward
    distributions and uses Bernstein inequality for building confidence intervals.
    For more information see the original UCB-V paper:
      https://www.sciencedirect.com/science/article/pii/S030439750900067X
    The algorithm has two parameters. The first vector of parameters indicates 
    the amount of exploration and the last term shows whether subsampling 
    is activated or not. The exploration term for arm i at time t is:
       sqrt(2 * estimated_variance * explore[t]) / #pulls of arm i) 
           + 3 * explore[t] / #pulls of arm i.

    Args:
      explore: The vector of exploration terms that appears in the definition of
        the UCB-V algorithm.
      sub_arm: Number of arms that are used. If None, it means all arms are used.

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """
    if sub_arm is None:
      sub_arm = self.k
    else:
      sub_arm = int(min(sub_arm, self.k))
  
    ind = random.sample(list(range(self.k)), sub_arm)   
    est_mean = np.zeros(self.k)
    pulls = np.zeros(self.T)
    regret = np.zeros(self.T)
    num_pulls = np.zeros(self.k)
    var = np.zeros(self.k)
    sum_sq = np.zeros(self.k)

    for t in range(self.T):
      if t <= sub_arm-1:
        a = ind[t]
        pulls[t] = a
        est_mean[a] = self.obs[t, a]
        sum_sq[a] = self.obs[t, a]**2
        var[a] = 0
      else:
        optimistic_rewards = est_mean[ind] + np.sqrt(
            2 * var[ind] * explore[t] / num_pulls[ind]) + \
            3 * explore[t] / num_pulls[ind]
        ind_max = optimistic_rewards.argmax()
        a = ind[ind_max]
        pulls[t] = a
        est_mean[a] = (self.obs[t, a] + est_mean[a] * num_pulls[a]) / (
            num_pulls[a] + 1)
        sum_sq[a] = sum_sq[a] + self.obs[t, a]**2
        var[a] = sum_sq[a] / (num_pulls[a] + 1) - est_mean[a]**2
      num_pulls[a] += 1
      regret[t] = max(self.means) - self.means[a]
    return regret, pulls, num_pulls


  def ucb(self, sub_arm=None):
    """Implements an asymptotically optimal UCB algorithm.

    This implements the asymptotically optimal UCB (as described in Chapter 8.1 of
    Lattimore & Szepesvari).

    Args:
      sub_arm: Number of arms that are used. If None, it means all arms are used.

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """

    explore = 2 * np.ones(self.T)
    explore_log = 1 + np.arange(1, self.T+1)*np.log(np.arange(1, self.T+1)**2)
    regret, pulls, num_pulls = self.generic_ucb(explore=explore,
                                                explore_log=explore_log,
                                                sub_arm=sub_arm)
    return regret, pulls, num_pulls


  def ucb_F(self, beta):
    """Implements the UCB-F algorithm.

    This implements the UCB-F algorithm, as explained in:
    https://papers.nips.cc/paper/2008/file/49ae49a23f67c759bf4fc791ba842aa2-Paper.pdf

    Args:
      beta: The beta parameter in the UCB-F algorithm.
      sub_arm: Number of arms that are used. If None, it means all arms are used.

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """
    if beta > 1:
      sub_arm = self.T**(beta/(beta+1))
    else:
      sub_arm = np.sqrt(self.T**beta)
    explore = 2 * np.log(10 * np.log(np.maximum(
        np.arange(1, self.T + 1), np.ones(self.T) * sub_arm
        )))
    regret, pulls, num_pulls = self.generic_ucb_v(explore=explore, 
                                                  sub_arm=sub_arm)
    return regret, pulls, num_pulls


  def ssucb(self, sub_arm=None):
    """Implements the SS-UCB algorithm.

    This function implements the SS-UCB algorithm as discussed in the paper.

    Args:
      sub_arm: Number of arms used. If None, the default is set to be sqrt(T).

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """
    if sub_arm is None:
      sub_arm = np.sqrt(self.T)
    regret, pulls, num_pulls = self.ucb(sub_arm=sub_arm)
    return regret, pulls, num_pulls


  def ts(self, beta_a=1, beta_b=1, sub_arm=None):
    """Implements the Thompson Sampling algorithm.

    This function implements the Thompson Sampling algorithm. Other than subsampling
    parameter (last parameter) which indicates how many arms to be used in execution,
    the other two parameters are only used when the rewards are Bernoulli. In this
    case, these parameters correspond to the parameters of our beta priors.

    Args:
      beta_a: The success rate of beta prior (only used for Bernoulli rewards).
      beta_b: The failure rate of beta prior (only used for Bernoulli rewards).
      sub_arm: Number of arms that are used. If None, it means all arms are used.

    Returns:
      regret: A vector containing the per-step regret values.
      pulls: A vector containing the pulls in different time-periods.
      num_pulls: The number of pulls for each of the arms.
    """
    if sub_arm is None:
      sub_arm = self.k
    else:
      sub_arm = int(min(sub_arm, self.k))

    if self.binary == 0:
      est_mean = 0.5 * np.ones(sub_arm) 
      inv_var = 16 * np.ones(sub_arm)
    else:
      S = beta_a * np.ones(sub_arm)
      F = beta_b * np.ones(sub_arm)
  
    ind = random.sample(list(range(self.k)), sub_arm)   
    pulls = np.zeros(self.T)
    regret = np.zeros(self.T)
    num_pulls = np.zeros(self.k)
    rand_vecs = np.zeros(sub_arm)

    for t in range(self.T):
      if(self.binary == 0):
        rand_vecs = est_mean+(1.0/np.sqrt(inv_var)) * np.random.randn(sub_arm)
        a = rand_vecs.argmax()
        arm = ind[a]
        pulls[t] = arm
        est_mean[a] = (self.obs[t, arm] + est_mean[a] * num_pulls[arm]) / (
            num_pulls[arm]+1)
        num_pulls[arm] += 1
        inv_var[a] += self.sigma**2
      else:
        rand_vecs = np.random.beta(S, F)
        a = rand_vecs.argmax()
        arm = ind[a]
        pulls[t] = arm
        if self.obs[t, arm] == 1:
          S[a] += 1
        else:
          F[a] += 1
        num_pulls[arm] += 1
      regret[t] = max(self.means) - self.means[arm]
    return regret, pulls, num_pulls