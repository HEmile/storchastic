Sampling and Inference
----------------------

Storchastic allows you to define stochastic computation graphs using an API that resembles generative stories.
It is designed with plug-and-play in mind: it is very easy to swap in different gradient estimation methods to compare
their performance on your task.

Small example
^^^^^^^^^^^^^
We return to the generative story from :ref:`Stochastic computation graphs`:

#. Compute :math:`d=a+b`

#. Sample :math:`e\sim \mathcal{N}(c+b, 1)`

#. Compute :math:`f=d\cdot e`

Let us first define the deterministic computations:

.. testsetup:: example

  import torch
  torch.manual_seed(0)

.. testcode:: example

  import torch
  a = torch.tensor(5.0, requires_grad=True)
  b = torch.tensor(-3.0, requires_grad=True)
  c = torch.tensor(0.23, requires_grad=True)
  d = a + b

Next, we want to sample from a normal distribution. We can define the normal distribution using PyTorch's distribution
code:

.. testcode:: example

  from torch.distributions import Normal
  normal_distribution = Normal(b + c, 1)

We will use Storchastic to sample from this distribution. We will use reparameterization for now:

.. testcode:: example

  from storch.method import Reparameterization
  method = Reparameterization("e")
  e = method(normal_distribution)

This samples a value from the normal distribution using reparameterization (or the pathwise derivative).
The :class:`storch.method.Reparameterization` class is a subclass of :class:`storch.method.Method`. Subclasses
implement functionality for sampling and gradient estimation, and can be subclassed to implement new methods.
Furthermore, :class:`storch.method.Method` subclasses :class:`torch.nn.Module`, which makes it easy for them
to become part of a PyTorch model.

Note that :class:`storch.method.Reparameterization` is initialized with the variable name "e". This is to initialize the
**plate** that corresponds to this sample. We will introduce plates later on. Also note that calling the method is
directly on the distribution to sample from. All methods are implemented so that they are called on just a :class:`torch.distributions.Distribution`.

We compute the output :math:`f` simply using:

.. testcode:: example

  f = d + e

Good. Now how to get the derivative? Storchastic requires you to register *cost nodes* using :func:`storch.add_cost`. These are
leave nodes that will be minimized. When all cost nodes are registered, :func:`storch.backward()` is used to estimate
the gradients:

.. testcode:: example

  import storch
  storch.add_cost(f, "f")
  storch.backward()
  print("haha yolo")

.. testoutput:: example
  :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

  hahadfg yolo xd

The second line register the cost node with the name "f", and the third computes the gradients, where PyTorch's automatic
differentiation is used for deterministic nodes, and Storchastic's gradient estimation methods for stochastic nodes.