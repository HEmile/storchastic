import storch
import torch
from torch.distributions import Bernoulli
from storch.method import RELAX

torch.manual_seed(0)

p = torch.tensor(0.5, requires_grad=True)
d = Bernoulli(p)
sample = RELAX("sample", in_dim=1)(d)
storch.add_cost(sample, "cost")
storch.backward()