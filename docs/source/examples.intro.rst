Introduction Example
====================
    .. code-block:: python
      import torch
      from torch.distributions import Normal
      from storch.method import Reparameterization
      import storch

      a = torch.tensor(5.0, requires_grad=True)
      b = torch.tensor(-3.0, requires_grad=True)
      c = torch.tensor(0.23, requires_grad=True)
      d = a + b

      normal_distribution = Normal(b + c, 1)

      method = Reparameterization("e")
      e = method(normal_distribution)

      f = d + e

      storch.add_cost(f, "f")
      storch.backward()
