from mmab import *
import argparse
import numpy as np

import scipy.stats
import scipy.special
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


def run_montecarlo(nsim, T_vals, k_vals, bern, sigma, beta_a, beta_b, save_results):
  '''Implements monte carlo simulations for comparing regret of algorithms.

  This function generates monte carlo instances that are used for comparing the regret
  of the algorithms discussed in the paper and returns regret and number of pulls of 
  arms. The function has the capability of simulating for several values of time horizon
  and number of arms. Please see the note on the shape of returns.
  
  Args:
    nsim: Number of simulations, i.e., monte carlo instances.
    T_vals: The list of values for time horizon.
    k_vals: The list of values for number of arms.
    bern: A boolean indicating whether to use Bernoulli or gaussian rewards.
    sigma: The standard deviation of noise (only used for gaussian rewards).
    beta_a: Success parameter of the beta prior.
    beta_b: Failure parameter of the beta prior.
    save_results: A boolean indicating whether to save the regret and number of pulls
      of various algorithms as .npy files.

  Returns:
    all_regret: A list of final (total) regret of algorithms. Each entry of the list is a
    numpy array of size nsim * number of different settings (specified by the length of
    T_vals and k_vals).
    pulls: Number of pulls across all arms reported only for the last configuration 
    given in T_vals and k_vals.  
  '''
  configs = len(T_vals)
  all_regret_greedy = np.zeros((nsim, configs))
  all_regret_ss_ucb = np.zeros((nsim, configs))
  all_regret_ucbf = np.zeros((nsim, configs))
  all_regret_ucb = np.zeros((nsim, configs))
  all_regret_ss_greedy = np.zeros((nsim, configs))
  all_regret_ts = np.zeros((nsim, configs))
  all_regret_ss_ts = np.zeros((nsim, configs))

  for j in range(configs):
    k = k_vals[j]
    T = T_vals[j]
    ## Regret Vectors
    regret_greedy = np.zeros(nsim)
    regret_ss_ucb = np.zeros(nsim)
    regret_ucbf = np.zeros(nsim)
    regret_ucb = np.zeros(nsim)
    regret_ss_greedy = np.zeros(nsim)
    regret_ts = np.zeros(nsim)
    regret_ss_ts = np.zeros(nsim)
    ## Pulls Vectors
    pulls_greedy = np.zeros((nsim, k))
    pulls_ucbf = np.zeros((nsim, k))
    pulls_ucb = np.zeros((nsim, k))
    pulls_ss_ucb = np.zeros((nsim, k))
    pulls_ss_greedy = np.zeros((nsim, k))
    pulls_ts = np.zeros((nsim, k))
    pulls_ss_ts = np.zeros((nsim, k))
    if bern == 0:
      greedy_sub_num_a = T**((beta_b+1)/3.0)
      ucb_sub_a = T**(beta_b/2.0)
      TS_sub_a = T**(beta_b/2.0)
    else:
      greedy_sub_num_a = T**(beta_b/2.0)
      ucb_sub_a = T**(beta_b/2.0)
      TS_sub_a = T**(beta_b/2.0)
    for itr in range(nsim):
      print('T=%d, k=%d, iteration = %d' % (T, k, itr))
      means = np.random.beta(beta_a, beta_b, k)
      ## Sorted version of means.
      vv = np.argsort(-means)
      A = MMAB(T=T, k=k, means=means, sigma=sigma, binary=bern) # Create an instance
      ## Greedy
      gr = A.greedy()
      gr_r = gr[0]
      regret_greedy[itr] = np.sum(gr_r)
      ## SS-Greedy
      gr_sub = A.greedy(sub_arm=greedy_sub_num_a)
      gr_sub_r = gr_sub[0]
      regret_ss_greedy[itr] = np.sum(gr_sub_r)
      ## UCB
      ucb = A.ucb()
      ucb_r = ucb[0]
      regret_ucb[itr] = np.sum(ucb_r)
      ## SS-UCB
      ucbs = A.ssucb(sub_arm=ucb_sub_a)
      ucbs_r = ucbs[0]
      regret_ss_ucb[itr] = np.sum(ucbs_r)
      ## UCBF
      ucbf = A.ucb_F(beta=beta_b)
      ucbf_r = ucbf[0]
      regret_ucbf[itr] = np.sum(ucbf_r)
      ## TS
      ts = A.ts(beta_a=beta_a, beta_b=beta_b)
      ts_r = ts[0]
      regret_ts[itr] = np.sum(ts_r)
      ## SS-TS
      ts_s = A.ts(beta_a=beta_a, beta_b=beta_b, sub_arm=TS_sub_a)
      ts_s_r = ts_s[0]
      regret_ss_ts[itr] = np.sum(ts_s_r)
      if j == configs-1:
        gr_np = gr[2]
        pulls_greedy[itr, :] = gr_np[vv]
        ##
        gr_sub_np = gr_sub[2]
        pulls_ss_greedy[itr, :] = gr_sub_np[vv]
        ##
        ucb_np = ucb[2]
        pulls_ucb[itr, :] = ucb_np[vv] 
        ##
        ucbs_np = ucbs[2]
        pulls_ss_ucb[itr, :] = ucbs_np[vv]
        ##
        ucbf_np = ucbf[2]
        pulls_ucbf[itr, :] = ucbf_np[vv]
        ##
        ts_np = ts[2]
        pulls_ts[itr, :] = ts_np[vv]
        ##
        ts_s_np = ts_s[2]
        pulls_ss_ts[itr, :] = ts_s_np[vv]

    regret = np.array([regret_greedy, 
                       regret_ss_greedy, 
                       regret_ucb, 
                       regret_ss_ucb, 
                       regret_ucbf, 
                       regret_ts, 
                       regret_ss_ts])
    pulls = np.array([pulls_greedy, 
                      pulls_ss_greedy, 
                      pulls_ucb, 
                      pulls_ss_ucb, 
                      pulls_ucbf, 
                      pulls_ts, 
                      pulls_ss_ts])

    if save_results == 1:
      if bern == 0:
        h = 'Norm_regret_T_{:d}_k_{:d}_a_{:.1f}_b_{:,.1f}_nsim_{:d}'.format(
            T_vals[j], k_vals[j], beta_a, beta_b, nsim)
        h = h.replace('.', '_')
        np.save(h + '.npy', regret)

        h = 'Norm_pulls_T_{:d}_k_{:d}_a_{:.1f}_b_{:,.1f}_nsim_{:d}'.format(
            T_vals[j], k_vals[j], beta_a, beta_b, nsim)
        h = h.replace('.', '_')
        np.save(h + '.npy', pulls)
      else:
        h = 'Bern_regret_T_{:d}_k_{:d}_a_{:.1f}_b_{:,.1f}_nsim_{:d}'.format(
            T_vals[j], k_vals[j], beta_a, beta_b, nsim)
        h = h.replace('.', '_')
        np.save(h + '.npy', regret)
      
        h = 'Bern_pulls_T_{:d}_k_{:d}_a_{:.1f}_b_{:,.1f}_nsim_{:d}'.format(
            T_vals[j], k_vals[j], beta_a, beta_b, nsim)
        h = h.replace('.', '_')
        np.save(h + '.npy', pulls)
      
    all_regret_greedy[:, j] = regret_greedy
    all_regret_ss_greedy[:, j] = regret_ss_greedy
    all_regret_ucb[:, j] = regret_ucb
    all_regret_ss_ucb[:, j] = regret_ss_ucb
    all_regret_ucbf[:, j] = regret_ucbf
    all_regret_ts[:, j] = regret_ts
    all_regret_ss_ts[:, j] = regret_ss_ts


  all_regret = np.array([all_regret_greedy, 
                         all_regret_ss_greedy, 
                         all_regret_ucb, 
                         all_regret_ss_ucb, 
                         all_regret_ucbf, 
                         all_regret_ts, 
                         all_regret_ss_ts])

  if save_results == 1:
    if bern == 0:
      h = 'Norm_all_regret_T_{:d}_k_{:d}_a_{:.1f}_b_{:,.1f}_nsim_{:d}'.format(
          T_vals[j], k_vals[j], beta_a, beta_b, nsim)
      h = h.replace('.', '_')
      np.save(h + '.npy', list([all_regret, T_vals, k_vals, beta_a, beta_b, nsim]))
    else:
      h = 'Norm_all_regret_T_{:d}_k_{:d}_a_{:.1f}_b_{:,.1f}_nsim_{:d}'.format(
          T_vals[j], k_vals[j], beta_a, beta_b, nsim)
      h = h.replace('.', '_')
      np.save(h + '.npy', list([all_regret, T_vals, k_vals, beta_a, beta_b, nsim]))
      
      
  return all_regret, pulls


def plot_results(T_vals, k_vals, regret, pulls, bern, beta_a, beta_b, save_plots):
  '''Generates regret and profile of pulls plot.

  This function generates the boxplots of regret and also the pulls vs the quantile
  of the (mean of) arms. The use of this function requires the plotly package.

  Args:
    T_vals: The list of values for time horizon.
    k_vals: The list of values for number of arms.
    regret: The list of final regret values for different configs defined in T_vals and
      k_vals.
    pulls: The list of pulls for different algorithms for the last config defined in
      T_vals and k_vals.
    bern: A boolean indicating whether to use Bernoulli or gaussian rewards.
    beta_a: Success parameter of the beta prior.
    beta_b: Failure parameter of the beta prior.
    save_plots: A boolean indicating whether to save the plots as png files or not.
  '''
  num_divs = 10

  z = max(np.floor(k_vals[-1] / num_divs), 1) + 1
  vals = np.arange(0, k_vals[-1], int(z))

  num_pts = regret[0].shape[1]
  niter = regret[0].shape[0]

  NUM_COLORS = 7
  MARKERS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle', 'pentagon', 
             'hexagram', 'star']
  legends = ['Greedy', 'SS-Greedy', 'UCB', 'SS-UCB', 'UCB-F', 'TS', 'SS-TS']
  pts_inc = num_pts
  
  color_vals = np.array([[0, 0, 0], [31, 119, 180], [255, 127, 14], 
                         [44, 160, 44], [214, 39, 40], [148, 103, 189],
                         [227,119,194], [188,189,34], [23, 190, 207]])

  color_alph = np.zeros((color_vals.shape[0], 3))
  for i in range(color_vals.shape[0]):
    color_alph[i,:] = make_rgb_transparent(color_vals[i,:], [255, 255, 255], 0.3)    

  colors=['rgb(0,0,0)', 'rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)',
          'rgb(227, 119, 194)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
          
  x_leg = []
  for j in range(pts_inc):
    f = niter * ['T={:d}, k={:d}'.format(
      T_vals[j-pts_inc+num_pts], k_vals[j+num_pts-pts_inc])]
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
        ticktext = [x_leg[1], x_leg[niter+1]],
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
    if bern == 0:
      h = 'Norm_regret_a_{:.1f}_b_{:,.1f}'.format(
        beta_a, beta_b)
      h = h.replace('.', '_')
      h = h + '.png'
      fig.write_image(h, scale = 1)
    else:
      h = 'Bern_regret_a_{:.1f}_b_{:,.1f}'.format(
        beta_a, beta_b)
      h = h.replace('.', '_')
      h = h + '.png'
      fig.write_image(h, scale = 1)
      
  tickz = []
  for j in range(num_divs):
    if j == 0:
      h = 'Top {:.0%} Arms'.format((j+1)/num_divs)
    else:
      h = '{:.0%}-{:.0%}'.format(j/num_divs, (j+1)/num_divs)
  tickz.append(h)

  h = int(np.floor(k_vals[-1]/num_divs))

  pulls_avg = np.zeros((num_divs, NUM_COLORS))
  pulls_std = np.zeros((num_divs, NUM_COLORS))
  for i in range(NUM_COLORS):
    for j in range(num_divs):
      z = np.arange(j*h,(j+1)*h)
      pulls_avg[j, i] = np.mean(pulls[i][:,z])
      pulls_std[j, i] = np.std(np.mean(pulls[i][:,z], axis = 1))/np.sqrt(niter)

  fig1 = go.Figure()


  for i in range(NUM_COLORS):
    fig1.add_trace(go.Scatter(
       y = np.log(pulls_avg[:, i]),
       x = np.arange(1,num_divs+1),
       name = legends[i],
       marker_symbol = i,
       marker_size = 16,
       marker_color = colors[i],
       mode = 'lines + markers',
       error_y = dict(
          type='data',
          array=np.log(pulls_avg[:,i]+2*pulls_std[:,i]) - np.log(pulls_avg[:,i]),
          arrayminus=np.log(pulls_avg[:,i]) - np.log(pulls_avg[:,i]-2*pulls_std[:,i]),
          visible=True,
          width=4
          )
       )
   )
  fig_title = 'Profile of pulls for T={:d} and k={:d}'.format(T_vals[-1], k_vals[-1])
  fig1.update_layout(
    title=dict(
    text = fig_title,
    y = 0.95,
    x = 0.5,
    xanchor = 'center',
    yanchor = 'top',
    font = dict(
        family ='sans-serif',
        size = 25,
        color = 'black',
        ),
    ),
    yaxis=dict(
        showgrid=True,
        zeroline=True,
        gridcolor='rgb(127, 127, 127)',
        gridwidth=1,
        zerolinecolor='rgb(0, 0, 0)',
        zerolinewidth=4,
        title = dict(
            text = r'Log Average Pulls',
            font = dict(
                family = 'sans-serif',
                size = 35,
                color = 'black'),
            ),
    ),
    width=1200,
    height=1200,
    font=dict(
    family='sans-serif',
    size=34,
    color='black',
    ),
    legend=dict(
        x=1,
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
        ticktext = tickz,
        tickvals = np.arange(1,num_divs+1),
        tickmode = 'array',
        tickfont_size = 24,
    ),
    margin=dict(l=120, r=50, t=20, b=20),
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(255,255,255)',
  )
  fig1.show()
  if save_plots == 1:
    if bern == 0:
      h = 'Norm_pulls_a_{:.1f}_b_{:,.1f}'.format(
        beta_a, beta_b)
      h = h.replace('.', '_')
      h = h + '.png'
      fig1.write_image(h, scale = 1)
    else:
      h = 'Bern_pulls_a_{:.1f}_b_{:,.1f}'.format(
        beta_a, beta_b)
      h = h.replace('.', '_')
      h = h + '.png'
      fig1.write_image(h, scale = 1)
  return


def __main__():
  parser = argparse.ArgumentParser(description='Running montecarlo simulations.')

  parser.add_argument('-T', type=int, nargs='+', 
                      help='<Required> Time horizon values', required=True)
  parser.add_argument('-k', type=int, nargs='+', 
                      help='<Required> Number of arms values', required=True)
  parser.add_argument('--nsim', type=int, help='Number of runs.', default=400)
  parser.add_argument('--seed', type=int, help='Random seed.', default=1252)
  parser.add_argument('--reward', type=str, help='Reward model.', default='NORMAL',
                      choices=['BERN', 'NORMAL'])
  parser.add_argument('--noise', type=float, 
                      help='Standard deviation of noise (for normal rewards).', 
                      default=1.0)    
  parser.add_argument('--beta_a', type=float, 
                      help='Success rate of the beta prior.', default=1.0)
  parser.add_argument('--beta_b', type=float, 
                      help='Failure rate of the beta prior.', default=1.0)                    
  parser.add_argument('--save_results', type=int, 
                      help='Whether to the results as numpy arrays.', 
                      default=1, choices=[0, 1])
  parser.add_argument('--save_plots', type=int, 
                      help='Whether to save the plots.', default=1, choices=[0, 1])                      

  args = parser.parse_args()
  
  np.random.seed(args.seed)
  bern = 0
  if args.reward=='BERN':
    bern = 1
  regret, pulls = run_montecarlo(nsim=args.nsim, 
                                 T_vals=args.T, 
                                 k_vals=args.k,
                                 bern=bern, 
                                 sigma=args.noise, 
                                 beta_a=args.beta_a, 
                                 beta_b=args.beta_b, 
                                 save_results=args.save_results)
  
  plot_results(T_vals=args.T, 
               k_vals=args.k,
               regret=regret, 
               pulls=pulls, 
               bern=bern, 
               beta_a=args.beta_a, 
               beta_b=args.beta_b, 
               save_plots=args.save_plots)


if __name__ == '__main__':
    __main__()
