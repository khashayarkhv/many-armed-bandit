# The Unreasonable Effectiveness of Greedy Algorithms for Multi-Armed Bandits with Many Arms

This repository contains codes for paper: [The Unreasonable Effectiveness of Greedy Algorithms for Multi-Armed Bandits with Many Arms](https://arxiv.org/abs/2002.10121) by Mohsen Bayati, Nima Hamidi, Ramesh Johari, and Khashayar Khosravi.

## File Descriptions

The scripts containing the implementation of various algorithms discussed in the paper are divided into four files discussed in the following:

* `mmab.py`: this class contains the implementation of algorithms discussed in the paper for the <i>stochastic</i> multi-armed bandit problem.

* `cmmab.py`: this class contains the implementation of algorithms discussed in the paper for the <i>contextual</i> multi-armed bandit problem.

* `montecarlo.py`: this file runs the monte carlo simulations for the stochastic case and generates the plots for the regret and profile of pulls.
 
* `contextual_montecarlo_datatest.py`: this file runs the monte carlo simulations for the contextual case using the Letter Recognition Dataset and generates the plots for the regret.

## Requirements

Requires python 3, numpy, scipy, and plotly (with orca for static image export).

## Datasets

The paper mainly relies on synthetic simulations, but there exist one simulation using the Letter Recognition Dataset. This dataset is publicly available, but please include the proper citation (described in the link below) if you wish to use this dataset: <a href="https://archive.ics.uci.edu/ml/datasets/Letter+Recognition"> Letter Recognition Dataset</a>.

## Citation

If you wish to use our repository in your work, please cite our paper:

Mohsen Bayati, Nima Hamidi, Ramesh Johari, and Khashayar Khosravi. **The Unreasonable Effectiveness of Greedy Algorithms for Multi-Armed Bandit with Many Arms**, arXiv preprint arXiv:2002.10121

A shorter version of our paper appeared at <a href="https://proceedings.neurips.cc/paper/2020/file/12d16adf4a9355513f9d574b76087a08-Paper.pdf"> NeurIPS 2020</a>.

BibTex:

```
@article{bayati2020unreasonable,
  title={The Unreasonable Effectiveness of Greedy Algorithms in Multi-Armed Bandit with Many Arms},
  author={Bayati, Mohsen and Hamidi, Nima and Johari, Ramesh and Khosravi, Khashayar},
  journal={arXiv preprint arXiv:2002.10121},
  year={2020}
}
```

Any question about the scripts can be directed to the authors <a href = "mailto: khashayar.khv@gmail.com"> via email</a>.

# Generating the figures in the paper

For generating the figures in the paper execute the following codes:

* Figure 1: `python3 montecarlo.py -T 20000 20000 -k 1000 3000`.

* Figure 2: `python3 contextual_montecarlo_datatest.py -T 8000 8000 -k 300 300 -d 2 6`.

* Figure 3: `python3 contextual_montecarlo_datatest.py -T 8000 8000 -k 8 8 -d 2 6`.
