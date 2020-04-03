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


def experiment(method):
    optim.zero_grad()
    b = Bernoulli(logits=eta.repeat(3))
    x = method(b)
    cost = torch.sum((x - p) ** 2, -1)
    storch.add_cost(cost, "cost")
    storch.backward()
    return eta.grad.clone()


print("True gradient", experiment(Expect("x")))

for n in range(2, 9):
    _method = ScoreFunction("x", n, baseline_factory="batch_average")
    gradients = []
    for i in range(1000):
        gradients.append(experiment(_method).unsqueeze(0))
    print(torch.cat(gradients).var(0))
