Storch Tensors
--------------

To keep track of the stochastic computation graph, Storchastic returns wrapped :class:`torch.Tensor` that are subclasses of
:class:`storch.Tensor`. This wrapper contains information that allows Storchastic to analyse the computation graph
during inference to properly estimate gradients. Furthermore, :class:`storch.Tensor` contains plate information that allows
for automatic broadcasting with other :class:`storch.Tensor` objects with different plate information.

.. autoclass:: storch.Tensor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: storch.StochasticTensor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: storch.CostTensor
   :members:
   :undoc-members:
   :show-inheritance:

Note: IndependentTensor might be removed in the future.

.. autoclass:: storch.tensor.IndependentTensor
   :members:
   :undoc-members:
   :show-inheritance: