import storch
from torch.distributions import Normal, Categorical
import torch
from storch.method import ScoreFunction

torch.manual_seed(0)

mu_prior = torch.tensor([2., -3.], requires_grad=True)
theta = torch.tensor([4., 5])


@storch.deterministic
def add(s1, s2, mu):
    return s2 + s1 * mu


@storch.stochastic
def white_noise(mu):
    return storch.sample(Normal(mu, 1), method=ScoreFunction(), n=1), storch.sample(Normal(-mu, 1), n=3)


@storch.cost
def loss(v):
    return torch.nn.MSELoss(reduction="none")(v, theta).mean(dim=-1)


mu = storch.sample(Normal(mu_prior, 1), n=4)
# k = storch.sample(Categorical(probs=[0.5, 0.5]))

agg_v = 0.
for i in range(2):
    s1, s2 = white_noise(mu)
    add_1 = add(s1, s2, mu)
    agg_v = add(agg_v, add_1, mu)
    loss(agg_v)

storch.backward(debug=True, keep_grads=True)
print(mu._accum_grads)
print(s1._accum_grads)
print(s2._accum_grads)
# storch.backwards()
