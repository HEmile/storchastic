import storch
import torch
from torch.distributions import Bernoulli
from storch.method import GumbelSoftmax

torch.manual_seed(0)

p = torch.tensor(0.5, requires_grad=True)

for i in range(10000):
    sample = GumbelSoftmax(f"sample_{i}")(Bernoulli(p))
    storch.add_cost(sample, f"cost_{i}")

storch.backward()
print("Finished")