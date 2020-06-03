Sampling and Inference
----------------------

Storchastic allows you to define stochastic computation graphs using an API that resembles generative stories.
It is designed with plug-and-play in mind: it is very easy to swap in different gradient estimation methods to compare
their performance on your task.

Sampling
^^^^^^^^
We return to the generative story from :ref:`Stochastic computation graphs`:

#. Compute :math:`d=a+b`

#. Sample :math:`e\sim \mathcal{N}(c+b, 1)` [#f1]_

#. Compute :math:`f=d\cdot e`

Let us first define the deterministic computations:

.. code-block:: python

  import torch
  a = torch.tensor(5.0, requires_grad=True)
  b = torch.tensor(-3.0, requires_grad=True)
  c = torch.tensor(0.23, requires_grad=True)
  d = a + b

Next, we want to sample from a normal distribution. We can define the normal distribution using PyTorch's distribution
code:

.. code-block:: python

  from torch.distributions import Normal
  normal_distribution = Normal(b + c, 1)

Next, we use Storchastic to sample from this distribution. We will use reparameterization for now:

.. code-block:: python

  from storch.method import Reparameterization
  method = Reparameterization("e")
  e = method(normal_distribution)

This samples a value from the normal distribution using reparameterization (or the pathwise derivative).
The :class:`storch.method.Reparameterization` class is a subclass of :class:`storch.method.Method`.