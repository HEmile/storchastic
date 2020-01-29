import storch
from torch.distributions import Normal
import torch

mu_prior = torch.tensor([1., 0.2, 0.3], requires_grad=True)
theta = torch.tensor([-5., 4.2, 4.3])


@storch.stochastic
def white_noise(mu):
    return storch.sample(Normal(mu, 1)), storch.sample(Normal(-mu, 1))


@storch.deterministic
def add(cum, s1, s2):
    return s2 * cum + s1


@storch.cost
def loss(v):
    return torch.nn.MSELoss()(v, theta)

mu = storch.sample(Normal(mu_prior, 1))

agg_v = 0.
for i in range(3):
    s1, s2 = white_noise(mu)
    agg_v = add(agg_v, s1, s2)
    print(agg_v)
c = loss(agg_v)
storch.walk_graph(c)
# storch.backwards()