"""
Based on Liu et al 2019, Kool et al 2020
"""

import torch
import storch
from torch.distributions import Bernoulli
from storch.method import ScoreFunction, Expect

p = torch.tensor([0.6, 0.51, 0.48])
eta = torch.tensor(-4.0, requires_grad=True)
optim = torch.optim.SGD([eta], lr=1.0)


def experiment(method, n):
    optim.zero_grad()
    b = Bernoulli(logits=eta.repeat(3))
    x = method("x", b, n)
    cost = torch.sum((x - p) ** 2, -1)
    storch.add_cost(cost, "cost")
    storch.backward()
    return eta.grad.clone()


print("True gradient", experiment(Expect(), 1))
_method = ScoreFunction(baseline_factory="batch_average")
for n in range(2, 9):
    gradients = []
    for i in range(1000):
        gradients.append(experiment(_method, n).unsqueeze(0))
    print(torch.cat(gradients).var(0))
