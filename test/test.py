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
    mult1 = s2 * mu
    add_1 = s1 + mult1
    agg_v = add_1 + agg_v
    # print(agg_v[..., 1])
    print(agg_v.max(dim=-1))
    print(agg_v.mean(dim=-1))
    print(agg_v.view((-1, 8)))
    loss(agg_v)

storch.backward(debug=False, keep_grads=True)
# print(mu._accum_grads)
# print(s1._accum_grads)
# print(s2._accum_grads)
