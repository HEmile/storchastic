"""
Based on Liu et al 2019, Kool et al 2020
"""

import torch
import storch
from torch.distributions import Bernoulli
from storch.method import ScoreFunction, Expect, UnorderedSetEstimator

p = torch.tensor([0.6, 0.51, 0.48])
eta = torch.tensor(-0.0, requires_grad=True)
optim = torch.optim.SGD([eta], lr=1.0)


def experiment(method):
    optim.zero_grad()
    b = Bernoulli(logits=eta.repeat(3))
    x = method(b)
    cost = torch.sum((x - p) ** 2, -1)
    storch.add_cost(cost, "cost")
    storch.backward()
    return eta.grad.clone()


true_grad = experiment(Expect("x"))
print("True gradient", true_grad)

for n in range(2, 9):
    # _method = ScoreFunction("x", n, baseline_factory="batch_average")
    _method = UnorderedSetEstimator("x", n)
    gradients = []
    for i in range(1000):
        gradients.append(experiment(_method).unsqueeze(0))
    gradients = torch.cat(gradients)
    print(n)
    print("variance", gradients.var(0))
    print("bias", torch.sum((gradients.mean(0) - true_grad) ** 2))
