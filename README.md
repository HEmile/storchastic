![](logo.png)




## General Stochastic Automatic Differentiation for Pytorch
[![Documentation Status](https://readthedocs.org/projects/storchastic/badge/?version=latest)](https://storchastic.readthedocs.io/en/latest/?badge=latest)

- [Documentation](https://storchastic.readthedocs.io/en/latest/)
- [Read the paper](https://arxiv.org/abs/2104.00428)

**Storchastic** is a PyTorch library for stochastic gradient estimation in Deep Learning [1]. Many state of the art deep learning
models use gradient estimation, in particular within the fields of Variational Inference and Reinforcement Learning.
While PyTorch computes gradients of deterministic computation graphs automatically, it will not estimate
gradients on **stochastic computation graphs** [2].

With Storchastic, you can easily define any stochastic deep learning model and let it estimate the gradients for you. 
Storchastic provides a large range of gradient estimation methods that you can plug and play, to figure out which one works
best for your problem. Storchastic provides automatic broadcasting of sampled batch dimensions, which increases code
readability and allows implementing complex models with ease.

When dealing with continuous random variables and differentiable functions, the popular reparameterization method [3] is usually
very effective. However, this method is not applicable when dealing with discrete random variables or non-differentiable functions.
This is why Storchastic has a focus on gradient estimators for discrete random variables, non-differentiable functions and
sequence models.


[Documentation on Read the Docs.](https://storchastic.readthedocs.io/en/latest/)

[Example: Discrete Variational Auto-Encoder](examples/vae/train.py)

## Installation
In your virtual Python environment, run
`pip install storchastic`

**Requires** Pytorch 1.**8** and [Pyro](http://pyro.ai). The code is build using Python 3.8.

## Algorithms
Feel free to create an issue if an estimator is missing here.
- Reparameterization [1, 3]
- Score Function (REINFORCE) with Moving Average baseline [1, 4]
- Score Function with Batch Average Baseline [5, 6]
- Expected value for enumerable distributions
- (Straight through) Gumbel Softmax [7, 8]
- LAX, RELAX [9] 
- REBAR [10]
- REINFORCE Without Replacement [6]
- Unordered Set Estimator [13]
- ARM [15]
- Rao-Blackwellized REINFORCE [12]

### In development
- Memory Augmented Policy Optimization [11]

### Planned
- Measure valued derivatives [1, 14]
- Automatic Credit Assignment [16]
- ...

## References
- [1] [Monte Carlo Gradient Estimation in Machine Learning](https://arxiv.org/abs/1906.10652), Mohamed et al, 2019
- [2] [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254), Schulman et al, NeurIPS 2015
- [3] [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), Kingma and Welling, ICLR 2014
- [4] [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link-springer-com.vu-nl.idm.oclc.org/article/10.1007/BF00992696), Williams, Machine Learning 1992
- [5] [Variational inference for Monte Carlo objectives](https://arxiv.org/abs/1602.06725), Mnih and Rezende, ICML 2016
- [6] [Buy 4 REINFORCE Samples, Get a Baseline for Free!](https://openreview.net/pdf?id=r1lgTGL5DE), Kool et al, ICLR Workshop dlStructPred 2019
- [7] [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144), Jang et al, ICLR 2017
- [8] [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712), Maddison et al, ICLR 2017
- [9] [Backpropagation through the Void: Optimizing control variates for black-box gradient estimation](https://arxiv.org/abs/1711.00123), Grathwohl et al, ICLR 2018
- [10] [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/abs/1703.07370), Tucker et al, NeurIPS 2017
- [11] [Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing](https://arxiv.org/abs/1807.02322), Liang et al, NeurIPS 2018
- [12] [Rao-Blackwellized Stochastic Gradients for Discrete Distributions](https://arxiv.org/abs/1810.04777), Liu et al, ICML 2019
- [13] [Estimating Gradients for Discrete Random Variables by Sampling without Replacement](https://openreview.net/forum?id=rklEj2EFvB), Kool et al, ICLR 2020
- [14] [Measure-Valued Derivatives for Approximate Bayesian Inference](http://bayesiandeeplearning.org/2019/papers/76.pdf), Rosca et al, Workshop on Bayesian Deep Learning (NeurIPS 2019)
- [15] [ARM: Augment-REINFORCE-Merge Gradient for Stochastic Binary Networks](https://arxiv.org/abs/1807.11143), Yin and Zhou, ICLR 2019
- [16] [Credit Assignment Techniques in Stochastic Computation Graphs](https://arxiv.org/abs/1901.01761), Weber et al, AISTATS 2019

## Cite
To cite Storchastic, please cite this preprint:
```
@article{van2021storchastic,
  title={Storchastic: A Framework for General Stochastic Automatic Differentiation},
  author={van Krieken, Emile and Tomczak, Jakub M and Teije, Annette ten},
  journal={arXiv preprint arXiv:2104.00428},
  year={2021}
}
```
