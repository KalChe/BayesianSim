# Multi-Fidelity Bayesian Optimization

This repo contains a simple simulation framework for cost-aware black box optimization with multiple noisy evaluators. The goal is to find high-value points on the unit n-dimensional sphere when evaluations have different costs and noise levels.

The setup is intentionally synthetic but meant to mirror real multi-fidelity settings (cheap + noisy vs expensive + accurate queries).

### What's going on?
There is an unknown function defined on the unit hypersphere with several Gaussian-style bumps and some global structure. You can only query it through approximators. Each approximator has:
1. A fixed cost ($c_i$)
2. A fixed noise level ($\sigma_i$)

You get a budget ($b$), and you must decide both **where** to sample and which approximator to use.

### Methods
* GreedySeq: Cheapest evaluator always and greedily selects the best looking points
* GP (UCB): A learned Gaussian process regression with a UCB-style acquisition that trades between value, uncertainty, and cost
* Thompson: Thompson sampling from the GP posterior

### To Run It
```bash
pip install numpy pandas matplotlib scikit-learn tqdm
python MultiFidelityBO.py
```

#### Why does this exist?
This is mostly for benchmarking ideas around adaptive querying with cost constraints, and for experimenting with multi-fidelity BO / bandit-style setups.

Author: Kalyan Cherukuri 

[Reach me!](mailto:kcherukuri@imsa.edu)
