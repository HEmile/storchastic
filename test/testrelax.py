import storch
import torch
from torch.distributions import Bernoulli, OneHotCategorical
from storch.method import RELAX, REBAR, ARM

torch.manual_seed(0)

p = torch.tensor(0.5, requires_grad=True)
d = Bernoulli(p)
sample = RELAX("sample", in_dim=1)(d)
# sample = ARM('sample', n_samples=10)(d)
storch.add_cost(sample, "cost")
storch.backward()

method = REBAR("test", n_samples=1)
x = torch.Tensor([[0.2, 0.4, 0.4], [0.5, 0.1, 0.4], [0.2, 0.2, 0.6], [0.15, 0.15, 0.7]])
qx = OneHotCategorical(x)
print(method(qx))