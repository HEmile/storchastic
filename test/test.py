import storch
from torch.distributions import Normal, Categorical
import torch

mu_prior = torch.tensor([2., -3.], requires_grad=True)
theta = torch.tensor(4.)


@storch.deterministic
def add(s1, s2):
    return s2 + s1


@storch.stochastic
def white_noise(mu):
    return add(storch.sample(Normal(mu, 1)), storch.sample(Normal(-mu, 1)))


@storch.cost
def loss(v):
    return torch.nn.MSELoss()(v, theta)


mu = storch.sample(Normal(mu_prior, 1))
# k = storch.sample(Categorical(probs=[0.5, 0.5]))

agg_v = 0.
for i in range(2):
    s1 = white_noise(mu)
    agg_v = add(agg_v, s1)
c = loss(agg_v)
storch.print_graph(c)
# storch.backwards()
