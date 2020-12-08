from cmmab import *
import argparse
import pandas as pd
import numpy as np
import scipy.special
import scipy.stats as sp
import plotly.graph_objs as go

def make_rgb_transparent(rgb, bg_rgb, alpha):
  '''Returns an RGB vector of values with given transparency level and background.
  
  This function is used for generating colors that are transparent with the background. 
  It has a similar functionality compared to alpha option in other libraries. The only
  difference is that it returns the rgb values of the transparent color.
  
  Args:
    rgb: The list rgb values for the original color(s).
    bg_rgb: The list of rgb values for the background(s).
    alpha: A number between 0 and 1 indicating the transparency level.
  Returns:
    rgb values for the transparent (mixed) colors.
  '''
  return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, bg_rgb)]
  

def draw_uniform(num_arms, dimension, norm=2):
  """Draws samples from unit ball specified by norm.

  This function draws arm parameters for any given dimension and the choice of norm being
  l-2 or l-infinity.

  Args:
    num_arms: Number of arms needed.
    dimension: Dimension of arms.
    norm: An l-2 or l-infinity norm to be used.
  Return:
    means: The vector of arm parameters drawn randomly from unit l-2 or l-infinity ball.
  """
  if norm == 'inf':
    means = 2 * np.random.rand(dimension, num_arms) - 1
  else:
    means = np.zeros((dimension, num_arms))
    ct = 0
    while ct < num_arms:
      h = 2 * np.random.rand(dimension)-1
      if np.linalg.norm(h, norm) <= 1:
        means[:, ct] =  h
        ct += 1
  return means


def run_montecarlo_contextual(nsim, T_vals, k_vals, d_vals, covariates, labels, 
                              true_labels, sigma, run_subs, normalize_mean_var, 
                              save_results):
  '''Implements monte carlo simulations for comparing regret of algorithms.

  This function generates monte carlo instances that are used for comparing the regret
  of the algorithms discussed in the paper and returns regret and number of pulls of 
  arms. The function has the capability of simulating for several values of time horizon
  and number of arms. Please see the note on the shape of returns.
  
  Args:
    nsim: Number of simulations, i.e., monte carlo instances.
    T_vals: The list of values for time horizon.
    k_vals: The list of values for number of arms.
    d_vals: The list of values for dimensions.
    covariates: The matrix of covariates.
    labels: The vector of labels.
    true_labels: Whether to use true labels or use the contexts to generate a linear model
      for labels (outcomes).
    sigma: Standard deviation of noise (only when true_labels is 0).
    run_subs: Whether to run algorithms that include subsampling step.
    normalize_mean_var: Whether to adjust covariates by centralizing using mean and 
      normalizing using standard deviation.

    save_results: A boolean indicating whether to save the regret and number of pulls
      of various algorithms as .npy files.

  Returns:
    all_regret: A list of final (total) regret of algorithms. Each entry of the list is a
    numpy array of size nsim * number of different settings (specified by the length of
    T_vals and k_vals).  
  '''
  configs = len(T_vals)
  all_regret_greedy = np.zeros((nsim, configs))
  all_regret_ucb = np.zeros((nsim, configs))
  all_regret_ts = np.zeros((nsim, configs))
  
  if run_subs==1:
    all_regret_ss_greedy = np.zeros((nsim, configs))
    all_regret_ss_ucb = np.zeros((nsim, configs))
    all_regret_ss_ts = np.zeros((nsim, configs))


  for j in range(configs):
    T = T_vals[j]
    k = k_vals[j]
    d = d_vals[j]
    subs_rate = 0.5
    greedy_sub_num_a = int(T**(subs_rate))
    ucb_sub_arm = int(T**(subs_rate))
    ts_sub_arm = int(T**(subs_rate))
    obs_norm = covariates
    if normalize_mean_var==1:
      for di in range(covariates.shape[1]):
        obs_norm[:,di] = (covariates[:, di] - np.mean(covariates[:, di])) / np.std(
          covariates[:, di]) 

    U, S, V = np.linalg.svd(obs_norm)
    dat = np.dot(np.dot(U[:, 0:d], np.diag(S[0:d])), np.transpose(V[0:d, 0:d]))

    regret_greedy = np.zeros(nsim)
    regret_ucb = np.zeros(nsim)
    regret_ts = np.zeros(nsim)
    
    if run_subs==1:
      regret_ss_greedy= np.zeros(nsim)
      regret_ss_ucb = np.zeros(nsim)
      regret_ss_ts = np.zeros(nsim)

    for itr in range(nsim):
      if true_labels==1:
        if k > np.amax(labels):
          print('Error: not enough classes')
        else:
          l = np.random.choice(np.amax(labels), np.amax(labels))
          h = [i for i,v in enumerate(labels) if v[l] <= k]
          indcs = np.random.choice(len(h), T)
          X = dat[h[indcs], :]
          Y = labels[h[z]]
      else:
        indcs = np.random.choice(dat.shape[0], T)
        X = dat[indcs, :]
      print('T=%d, k=%d, d=%d, iteration = %d' % (T, k, d, itr))
      means = draw_uniform(num_arms=k, 
                           dimension=d, 
                           norm=2)
      if true_labels==1:
        A = CMMAB(T, k, d, context = X, labels = Y)
      else:
        A = CMMAB(T, k, d, context = X, means = means, sigma=sigma)
      ## Greedy
      gr = A.greedy()
      gr_r = gr[0]
      regret_greedy[itr] = np.sum(gr_r)
      all_regret_greedy[itr, j] = regret_greedy[itr]
      ## Subsampled Greedy
      if run_subs==1:
        gr_sub = A.greedy(sub_arm=greedy_sub_num_a)
        gr_sub_r = gr_sub[0]
        regret_ss_greedy[itr] = np.sum(gr_sub_r)
        all_regret_ss_greedy[itr, j] = regret_ss_greedy[itr]
      ## OFUL
      oful = A.oful()
      oful_r = oful[0]
      regret_ucb[itr] = np.sum(oful_r)
      all_regret_ucb[itr, j] = regret_ucb[itr]
      ## SS-OFUL
      if run_subs==1:
        ucbs = A.oful(sub_arm=ucb_sub_arm)
        ucbs_r = ucbs[0]
        regret_ss_ucb[itr] = np.sum(ucbs_r)
        all_regret_ss_ucb[itr, j] = regret_ss_ucb[itr]
      ## TS
      ts = A.ts()
      ts_r = ts[0]
      regret_ts[itr] = np.sum(ts_r)
      all_regret_ts[itr, j] = regret_ts[itr]
      ## SS-TS
      if run_subs==1:
        ts_s = A.ts(sub_arm=ts_sub_arm)
        ts_s_r = ts_s[0]
        regret_ss_ts[itr] = np.sum(ts_s_r)
        all_regret_ss_ts[itr, j] = regret_ss_ts[itr]

    if run_subs==1:
      regret = np.array([regret_greedy, 
                        regret_ss_greedy, 
                        regret_ucb, 
                        regret_ss_ucb, 
                        regret_ts, 
                        regret_ss_ts])
    else:
      regret = np.array([regret_greedy, 
                        regret_ucb, 
                        regret_ts])

    if save_results == 1:
      if true_labels == 1:
        h = "Real_regret_T_{:d}_k_{:d}_d_{:d}_run_subs_{:d}_nsim_{:d}".format(
            T_vals[j], k_vals[j], d_vals[j], run_subs, nsim)
        h = h.replace(".", "_")
        np.save(h + ".npy", regret)

      else:
        h = "Semi_regret_T_{:d}_k_{:d}_d_{:d}_run_subs_{:d}_nsim_{:d}".format(
            T_vals[j], k_vals[j], d_vals[j], run_subs, nsim)
        h = h.replace(".", "_")
        np.save(h + ".npy", regret)
        
  if run_subs==1:
    all_regret = np.array([all_regret_greedy, 
                           all_regret_ss_greedy, 
                           all_regret_ucb, 
                           all_regret_ss_ucb, 
                           all_regret_ts, 
                           all_regret_ss_ts])
  else:
    all_regret = np.array([all_regret_greedy, 
                           all_regret_ucb, 
                           all_regret_ts])  

  if save_results == 1:
    if true_labels == 1:
      h = "Real_all_regret_T_{:d}_k_{:d}_d_{:d}_run_subs_{:d}_nsim_{:d}".format(
          T_vals[j], k_vals[j], d_vals[j], run_subs, nsim)
      h = h.replace(".", "_")
      np.save(h + ".npy", list([all_regret, T_vals, k_vals, d_vals, run_subs, nsim]))
    else:
      h = "Semi_all_regret_T_{:d}_k_{:d}_d_{:.1f}_nsim_{:d}".format(
          T_vals[j], k_vals[j], d_vals[j], run_subs, nsim)
      h = h.replace(".", "_")
      np.save(h + ".npy", list([all_regret, T_vals, k_vals, d_vals, run_subs, nsim]))
      
  return all_regret


def plot_results(T_vals, k_vals, d_vals, regret, true_labels, run_subs, save_plots):
  '''Generates regret plots.

  This function generates the boxplots of regret. This function requires the plotly 
  package for execution.

  Args:
    T_vals: The list of values for time horizon.
    k_vals: The list of values for number of arms.
    d_vals: The list of values for dimensions.
    regret: The list of final regret values for different configs defined in T_vals and
      k_vals.
    true_labels: Whether to the true labels have been used or not.
    run_subs: Whether to the algorithms that include subsampling step have been executed.
    save_plots: A boolean indicating whether to save the plots as png files or not.
  '''
  num_pts = regret[0].shape[1]
  nsim = regret[0].shape[0]
  pts_inc = num_pts
  if run_subs == 1:
    NUM_COLORS = 6
    MARKERS = ['circle', 'square', 'diamond', 'cross', 'triangle', 'pentagon', 
               'hexagram', 'star']
    legends = ['Greedy', 'SS-Greedy', 'UCB', 'SS-UCB', 'TS', 'SS-TS']
  
    color_vals = np.array([[0, 0, 0], [31, 119, 180], [255, 127, 14], 
                           [44, 160, 44], [148, 103, 189], [227, 119, 194]])
                         
    colors=['rgb(0,0,0)', 'rgb(31, 119, 180)', 'rgb(255, 127, 14)',
            'rgb(44, 160, 44)', 'rgb(148, 103, 189)', 'rgb(227, 119, 164)']
  else:
    NUM_COLORS = 3
    MARKERS = ['circle', 'diamond', 'triangle']
    legends = ['Greedy', 'UCB', 'TS']
  
    color_vals = np.array([[0, 0, 0], [255, 127, 14], [148, 103, 189]])
  
    colors=['rgb(0,0,0)', 'rgb(255, 127, 14)', 'rgb(148, 103, 189)']

  color_alph = np.zeros((color_vals.shape[0], 3))
  for i in range(color_vals.shape[0]):
      color_alph[i,:] = make_rgb_transparent(color_vals[i,:], [255, 255, 255], 0.3)    
     
  x_leg = []
  for j in range(pts_inc):
    f = nsim * ['T={:d}, k={:d}, d={:d}'.format(
      T_vals[j-pts_inc+num_pts], k_vals[j+num_pts-pts_inc], d_vals[j+num_pts-pts_inc])]
    x_leg += f

  fig = go.Figure()

  for i in range(NUM_COLORS):
    fig.add_trace(go.Box(
      y=regret[i][:,num_pts-pts_inc:].transpose().flatten(),
      x=x_leg,
      name=legends[i],
      fillcolor = 'rgb({:f},{:f},{:f})'.format(
        color_alph[i, 0], color_alph[i, 1], color_alph[i, 2]),
      marker=dict(
          color=colors[i],
          size=10,
          opacity=1,
          symbol = i
          ),
      showlegend = False,
      boxmean = True,
      boxpoints = 'outliers',
    ))
    fig.add_trace(go.Scatter(
        y=[0.9*np.max(regret)],
        x=[0.6],
        name=legends[i],
        mode='markers',
        marker_symbol=i,
        marker_size=16,
        marker_color=colors[i]
    ))
  fig.update_layout(
    autosize = False,
    yaxis=dict(
        showgrid=True,
        zeroline=True,
        gridcolor='rgb(127, 127, 127)',
        gridwidth=1,
        zerolinecolor='rgb(0, 0, 0)',
        zerolinewidth=3,
        title = dict(
            text = 'Regret',
            font = dict(
                family = 'sans-serif',
                size = 35,
                color = 'black'
                ),
            ),
    ),
    boxmode='group',
    width=1200,
    height=1200,
    font=dict(
    family='sans-serif',
    size=35,
    color='black',
    ),
    legend=dict(
        x=0.8,
        y=1,
        traceorder='normal',
        font=dict(
            family='sans-serif',
            size=35,
            color='black'
        ),
        bgcolor='white',
        bordercolor='Black',
        borderwidth=6,
    ),
    xaxis=dict(
        ticktext = [x_leg[1], x_leg[nsim+1]],
        tickvals = [0, 1],
        tickmode = 'array',
        tickfont_size = 30,
        scaleanchor = 'y',
        ticklen = 2,
    ),
    margin=dict(l=120, r=50, t=20, b=20),
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(255,255,255)',
    boxgap = 0.4,
    boxgroupgap = 0.1,
  )
  fig.show()
  if save_plots == 1:
    if true_labels == 1:
      h = 'Real_regret_run_subs_{:d}'.format(run_subs)
      h = h.replace('.', '_')
      h = h + '.png'
      fig.write_image(h, scale = 1)
    else:
      h = 'Semi_regret_run_subs_{:d}'.format(run_subs)
      h = h.replace('.', '_')
      h = h + '.png'
      fig.write_image(h, scale = 1)
  return


def __main__():
  parser = argparse.ArgumentParser(description='Running montecarlo simulations.')

  parser.add_argument('-T', type=int, nargs='+', 
                      help='<Required> Time horizon values', required=True)
  parser.add_argument('-k', type=int, nargs='+', 
                      help='<Required> Number of arms values', required=True)
  parser.add_argument('-d', type=int, nargs='+', 
                      help='<Required> Dimension values', required=True)
  parser.add_argument('--nsim', type=int, help='Number of runs.', default=50)
  parser.add_argument('--seed', type=int, help='Random seed.', default=1252)
  parser.add_argument('--true_labels', type=int, help='Use true label or not.', 
                      default=0)
  parser.add_argument('--run_subs', type=int, 
                      help='Run algorithms with subsampling or not.', default=1)
  parser.add_argument('--noise', type=float, 
                      help='Standard deviation of noise (when outcomes are generated).', 
                      default=1.0)     
  parser.add_argument('--normalize_mean_var', type=int, 
                      help='To center/standardize contexts using mean and std.', 
                      default=1, choices=[0, 1])              
  parser.add_argument('--save_results', type=int, 
                      help='Whether to the results as numpy arrays.', 
                      default=1, choices=[0, 1])
  parser.add_argument('--save_plots', type=int, 
                      help='Whether to save the plots.', default=1, choices=[0, 1])                      

  args = parser.parse_args()

  
  np.random.seed(args.seed)
  
  ## Preprocess Data
  my_data = pd.read_csv('dataset_6_letter.csv')
  my_data.convert_dtypes()
  for j in range(my_data.shape[0]):
    new_datafr = my_data['class'].copy(deep=True) 
    new_datafr[j] = ord(my_data['class'][j].lower())-97
  my_data['class'] = new_datafr
  cols = my_data.columns.values
  cov_cols = cols[cols!='class']
  covariates = np.array(my_data[cov_cols])
  labels = my_data['class']
  
  regret = run_montecarlo_contextual(nsim=args.nsim, 
                                     T_vals=args.T, 
                                     k_vals=args.k,
                                     d_vals=args.d,
                                     covariates=covariates,
                                     labels=labels,
                                     true_labels=args.true_labels, 
                                     sigma=args.noise, 
                                     run_subs=args.run_subs, 
                                     normalize_mean_var=args.normalize_mean_var, 
                                     save_results=args.save_results)
  
  
  plot_results(T_vals=args.T, 
               k_vals=args.k,
               d_vals=args.d,
               regret=regret, 
               true_labels=args.true_labels,
               run_subs=args.run_subs,
               save_plots=args.save_plots)


if __name__ == '__main__':
  __main__()