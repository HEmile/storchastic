
.. image:: images/logo.png
  :width: 600
  :alt: Storchastic logo.

**Storchastic** is a PyTorch library for stochastic gradient estimation in Deep Learning. Stochastic deep learning models
are becoming increasingly relevant. For example, they are commonly used in the fields of Variational
Inference and Reinforcement Learning. We can formalize stochastic models using so-called **stochastic computation graphs**.
While PyTorch computes gradients of deterministic computation graphs automatically, PyTorch will not automatically estimate
gradients on such stochastic graphs. This is because they require marginalization over the stochastic nodes in the graph,
which is usually intractable and needs to be estimated.

With Storchastic, you can easily define any stochastic deep learning model and let it estimate the gradients for you.
Storchastic provides a large range of gradient estimation methods that you can plug and play, to figure out which one works
best for your problem. Storchastic provides automatic broadcasting of sampled batch dimensions, which increases code
readability and allows implementing complex models with ease.


When dealing with continuous random variables and differentiable functions, the popular reparameterization method is usually
very effective. However, this method is not applicable when dealing with discrete random variables or non-differentiable functions.
This is why Storchastic has a focus on gradient estimators for discrete random variables, non-differentiable functions and
sequence models.

Mail e.van.krieken@vu.nl for any help or questions.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   intro
   license
   faq
   help
   examples

.. toctree::
   :maxdepth: 3
   :caption: API documentation:

   storch.tensor
   storch.plate
   storch.method
   storch.sampling
   storch


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
