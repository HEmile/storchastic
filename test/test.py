import storch
from torch.distributions import Normal, Categorical
import torch
from storch.method import ScoreFunction

torch.manual_seed(0)

mu_prior = torch.tensor([2., -3.], requires_grad=True)
theta = torch.tensor([4., 5])

method = storch.method.Infer(Normal)
score_method = storch.method.ScoreFunction()


def loss(v):
    return torch.nn.MSELoss(reduction="none")(v, theta).mean(dim=-1)


mu = method("mu", Normal(mu_prior, 1), n=1)
# k = storch.sample(Categorical(probs=[0.5, 0.5]))

agg_v = 0.
s1 = 0.
for i in range(2):
    s1 = method("white_noise_1", Normal(mu, 1), n=2)
    s2 = method("white_noise_2", Normal(-mu + s1, 1), n=1)
    # plus = lambda a, b: a + b
    # plus = storch.deterministic(plus)
    agg_v = agg_v + s1 + s2 * mu
    storch.add_cost(loss(agg_v), "loss")

storch.backward(debug=True, accum_grads=False)
