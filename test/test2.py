import storch
from torch.distributions import Normal
import torch

mu = torch.tensor([1., 0.2, 84.3], requires_grad=True)
theta = torch.tensor([-5., 4.2, 4.3])


@storch.stochastic
def white_noise(mu):
    return storch.sample(Normal(mu, 1)), torch.tensor([4., 3.])
print(white_noise(mu)[0])

# @storch.stochastic
# def white_noise(mu):
#     # Should crash: calls a stochastic context within a stochastic context
#     return white_noise(storch.sample(Normal(mu, 1)))
# s, c = white_noise(mu)


a = torch.tensor(0.4, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
c = torch.tensor(0.6, requires_grad=True)

# d = a + b
# e = d * c
# print(has_differentiable_path(d, a)) # True
# print(has_differentiable_path(d, c)) # False
# print(has_differentiable_path(e, d)) # True
# print(has_differentiable_path(d, e)) # False
# print(has_differentiable_path(e, c)) # True
# print(has_differentiable_path(e, b)) # True