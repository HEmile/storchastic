
Storchastic documentation
=========================
**Storchastic** is a PyTorch library for stochastic gradient estimation in Deep Learning. Many state of the art deep learning
models use gradient estimation, in particular within the fields of Variational Inference and Reinforcement Learning.
While PyTorch computes gradients of deterministic computation graphs automatically, it will not estimate
gradients on **stochastic computation graphs**.

With Storchastic, you can easily define any stochastic deep learning model and let it estimate the gradients for you.
Storchastic provides a large range of gradient estimation methods that you can plug and play, to figure out which one works
best for your problem. Storchastic provides automatic broadcasting of sampled batch dimensions, which increases code
readability and allows implementing complex models with ease.

When dealing with continuous random variables and differentiable functions, the popular reparameterization method is usually
very effective. However, this method is not applicable when dealing with discrete random variables or non-differentiable functions.
This is why Storchastic has a focus on gradient estimators for discrete random variables, non-differentiable functions and
sequence models.

Note: Documentation is currently under construction.

Mail e.van.krieken@vu.nl for any help or questions.

Guide
^^^^^
.. toctree::
   :maxdepth: 3
   :caption: Contents:

   license
   help
   storch
   discrete_vae



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
