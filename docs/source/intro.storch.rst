Sampling, Inference and Variance Reduction
------------------------------------------

Storchastic allows you to define stochastic computation graphs using an API that resembles generative stories.
It is designed with plug-and-play in mind: it is very easy to swap in different gradient estimation methods to compare
their performance on your task. In this tutorial, we apply gradient estimation to a simple and small problem.

.. role:: python(code)
   :language: python

Converting generative stories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We return to the generative story from :ref:`Stochastic computation graphs`:

#. Compute :math:`d=a+b`

#. Sample :math:`e\sim \mathcal{N}(c+b, 1)`

#. Compute :math:`f=d\cdot e^2`

This story is easily converted using the following code:

.. code-block:: python
  :linenos:
  :emphasize-lines: 13,14,15

  import torch
  import storch
  from torch.distributions import Normal
  from storch.method import Reparameterization, ScoreFunction

  def compute_f(n):
      a = torch.tensor(5.0, requires_grad=True)
      b = torch.tensor(-3.0, requires_grad=True)
      c = torch.tensor(0.23, requires_grad=True)
      d = a + b

      # Sample e from a normal distribution using reparameterization
      normal_distribution = Normal(b + c, 1)
      method = Reparameterization("e", n_samples=n)
      e = method(normal_distribution)

      f = d * e * e
      return f, c

Lines 10 and 17 represent the deterministic nodes. Lines 13-15 represent the stochastic node:
We sample a value from the normal distribution using reparameterization (or the pathwise derivative).
The :class:`storch.method.Reparameterization` class is a subclass of :class:`storch.method.Method`. Subclasses
implement functionality for sampling and gradient estimation, and you can subclass :class:`storch.method.Method` to
implement new methods gradient estimation methods. Furthermore, :class:`storch.method.Method` subclasses
:class:`torch.nn.Module`, which makes it easy for them to become part of a PyTorch model.

:class:`storch.method.Reparameterization` is initialized with the variable name "e". This is done to initialize the
**plate** that corresponds to this sample. We will introduce plates later on. Furthermore, they have an optional
`n_samples` option, which controls how many samples are taken from the normal distribution. Note that the method is called directly on the distribution (:class:`torch.distributions.Distribution`) to sample from.

Gradient estimation
^^^^^^^^^^^^^^^^^^^
Great. Now how to get the derivative with respect to :math:`c`? Storchastic requires you to register *cost nodes* using :func:`storch.add_cost`. These are
leave nodes that will be minimized. When all cost nodes are registered, :func:`storch.backward()` is used to estimate
the gradients:

.. code-block:: python

  >>> f, c = compute_f(1)
  >>> storch.add_cost(f, "f")
  >>> storch.backward()
  tensor(3.0209, grad_fn=<AddBackward0>)
  >>> c.grad
  tensor(-4.9160)

The second line registers the cost node with the name "f", and the third line computes the gradients, where PyTorch's automatic
differentiation is used for deterministic nodes, and Storchastic's gradient estimation methods for stochastic nodes.
:func:`storch.backward` returns the estimated value of the sum of cost nodes, which in this case is just :math:`f`.

We also show the estimated gradient with respect to :math:`c` (-4.9160). Note that this gradient is stochastic! Running
the code another time, we get -12.2537.

Computing gradient statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can estimate the mean and variance of the gradient as follows:

.. code-block:: python
  :lineno-start: 19

  n = 1
  gradient_samples = []
  for i in range(1000):
      f, c = compute_f(n)
      storch.add_cost(f, "f")
      storch.backward()
      gradient_samples.append(c.grad)
  gradients = storch.gather_samples(gradient_samples, "gradients")

.. code-block:: python

  >>> storch.variance(gradients, "gradients")
  Deterministic tensor(16.7321) Batch links: []
  >>> print(storch.reduce_plates(gradients, "gradients"))
  Deterministic tensor(-11.0195) Batch links: []

Alright, a few things to note. :func:`storch.gather_samples` is a function that takes a list of tensors that are (conditionally)
independent samples of some value, in this case the gradients. Like most other methods in Storchastic, it returns a
:class:`storch.Tensor`, in this case a :class:`storch.IndependentTensor`:

.. code-block:: python

  >>> type(gradients)
  <class 'storch.tensor.IndependentTensor'>

:class:`storch.Tensor` is a special "tensor-like" object which wraps a :class:`torch.Tensor` and includes extra metadata
to help with estimating gradients and keeping track of the plate dimensions. Plate dimensions are dimensions of the tensor
of which we know conditional independency properties. We can look at the plate dimensions of a :class:`storch.Tensor` using

.. code-block:: python

  >>> gradients.plates
  [('gradients', 1000, tensor(0.0010))]

The gradients tensor has one plate dimension with name "gradients" (as we defined using :func:`storch.gather_samples`).
As we simulated the gradient 1000 times, the size of the plate dimension is 1000. The third value is the **weight** of the
samples. In this case, samples are weighted identically (that is, the weight is 1/1000), which corresponds to a normal
monte carlo sample.

Note that we used the plate dimension name "gradients" in :python:`storch.variance(gradients, "gradients")`.
With this we mean that we compute the variance over the gradient plate dimension, which represent the
different independent samples of gradient estimates.

Reducing variance
^^^^^^^^^^^^^^^^^
Next, let us try to reduce the variance. A simple way to do this is to use more samples of :math:`e`.
In line 14 (:python:`method = Reparameterization("e", n_samples=n)`, we pass the amount of samples to use
for this method. Let's use 10 by setting line 19 to :python:`n = 10`, and compute the variance again:

.. code-block:: python

  >>> storch.variance(gradients, "gradients")
  Deterministic tensor(1.6388) Batch links: []

By using 10 times as many samples, we reduced the variance by (about) a factor 10. Note that we did
not have to change any other code but changing the value of n. Storchastic is designed so that all
(left-broadcastable!) code supports both using a single or multiple samples.
Using more samples is an easy way to reduce variance. Storchastic automatically parallelizes the
computation over the different samples, so that if your gpu has enough memory, there is (usually) almost
no overhead to using more samples, yet we can get better estimates of the gradient!

Using different estimators
^^^^^^^^^^^^^^^^^^^^^^^^^^
Storchastic is designed to make swapping in different gradient estimation as easy as possible. For instance, say we want
to use the score function instead of reparameterization. This is done as follows:

.. code-block:: python
  :lineno-start: 6
  :emphasize-lines: 9

  def compute_f(n):
      a = torch.tensor(5.0, requires_grad=True)
      b = torch.tensor(-3.0, requires_grad=True)
      c = torch.tensor(0.23, requires_grad=True)
      d = a + b

      # Sample e from a normal distribution using reparameterization
      normal_distribution = Normal(b + c, 1)
      method = ScoreFunction("e", n_samples=n, baseline_factory=None)
      e = method(normal_distribution)

      f = d * e * e
      return f, c

Note how we only changed the line (:python:`method = Reparameterization("e", n_samples=n)`) where we defined the gradient
estimation method to now create a :class:`storch.method.ScoreFunction` instead of :class:`storch.method.Reparameterization`.
Let's see the variance of this method (using 1 sample):

.. code-block:: python

  >>> storch.variance(gradients, "gradients")
  Deterministic tensor(748.1914) Batch links: []

Ouch, that really is much higher than using Reparameterization! While the score function is much more generally applicable
than reparameterization (as it can be used for discrete distributions and non-differentiable functions), it clearly has
a prohibitive large variance. Storchastic also has the :class:`storch.method.Infer` gradient estimation method,
which automatically applies reparameterization if possible and otherwise uses the score function.

Can we do something about the large variance? Using more samples is always an option.
To get the variance in the same ballpark as a single-sample reparameterization, we would need to use about 748.2/16.7
samples, or about n=45!

.. code-block:: python

  >>> storch.variance(gradients, "gradients")
  Deterministic tensor(17.0591) Batch links: []

Luckily, we can make efficient reuse of the multiple samples we take. Note how we set :python:`baseline_factory=None`
when defining the :class:`storch.method.ScoreFunction`. A baseline is a very common variance reduction method that
subtracts a value from the cost function to stabilize the gradient. A simple but effective one is the batch average
baseline (:class:`storch.method.baseline.BatchAverage`) that subtracts the average of the other samples. Simply change
:python:`ScoreFunction("e", n_samples=n, baseline_factory="batch_average")`. Let's use 20 samples:

.. code-block:: python

  >>> storch.variance(gradients, "gradients")
  Deterministic tensor(16.8761) Batch links: []

Sweet! We used fewer than halve of the samples, yet get a lower variance than before. For complicated settings where
reparameterization is not an option, strong variance reduction is unfortunately very important for efficient algorithms.

For full code of this example, go to :doc:`examples.intro`.