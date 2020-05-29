"""
Based on Yin and Zhou, ICLR 2019
"""

import torch
import storch
from torch.distributions import Bernoulli
from storch.method import ScoreFunction, Expect, UnorderedSetEstimator

p = torch.tensor([0.49, 0.499, 0.501, 0.51])
eta = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
optim = torch.optim.SGD([eta], lr=1.0)


def experiment(method):
    for i in range(2000):
        optim.zero_grad()
        b = Bernoulli(logits=eta)
        x = method(b)
        cost = torch.sum((x - p) ** 2, -1)
        storch.add_cost(cost, "cost")
        storch.backward()
        optim.step()
        if i % 100 == 0:
            print(eta)
        # return eta.grad.clone()


experiment(Expect("x"))
# experiment(UnorderedSetEstimator("x", 5))
# print("True gradient", true_grad)

# for n in range(2, 9):
#     # _method = ScoreFunction("x", n, baseline_factory="batch_average")
#     _method = UnorderedSetEstimator("x", n)
#     # gradients = []
#     for i in range(1000):
#         experiment(_method)
#     # gradients = torch.cat(gradients)
#     # print(n)
#     # print("variance", gradients.var(0))
#     # print("bias", torch.sum((gradients.mean(0) - true_grad) ** 2))
