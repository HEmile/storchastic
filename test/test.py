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
    print(mu.shape)
    return storch.sample(Normal(mu, 1), n=1), storch.sample(Normal(-mu, 1), n=3)


@storch.cost
def loss(v):
    return torch.nn.MSELoss(reduction="none")(v, theta).mean(dim=-1)


mu = storch.sample(Normal(mu_prior, 1), n=4)
# k = storch.sample(Categorical(probs=[0.5, 0.5]))

agg_v = 0.
for i in range(2):
    s1, s2 = white_noise(mu)
    print("s1", s1, "s2", s2)
    add_1 = add(s1, s2)
    agg_v = add(agg_v, add_1)
    print(agg_v)
    loss(agg_v)
# storch.backward(debug=True)
# storch.backwards()
