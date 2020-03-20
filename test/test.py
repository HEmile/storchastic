import storch
from torch.distributions import Normal, Categorical
import torch
from storch.method import ScoreFunction

torch.manual_seed(0)

mu_prior = torch.tensor([2.0, -3.0], requires_grad=True)
theta = torch.tensor([4.0, 5])

method = storch.method.Infer(Normal)
score_method = storch.method.ScoreFunction()
expect = storch.method.Expect()


def loss(v):
    return torch.nn.MSELoss(reduction="none")(v, theta).mean(dim=-1).mean(dim=-1)


mu = method("mu", Normal(mu_prior, 1), n=1)
k = expect(
    "k",
    Categorical(
        probs=torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.8, 0.1]], requires_grad=True)
    ),
)

agg_v = 0.0
s1 = 1.0
for i in range(2):
    k1 = k[:, 0]
    k2 = k[:, 1]
    # Take 2 samples of 3 independent 2-dimensional (as mu is 2-dimensional) multivariate gaussians.
    s1 = score_method("white_noise_1", Normal(mu + k1, 1), n=2, event_shape=3)
    # Reduce over the independent event dimension.
    additive = (s1 * k2.unsqueeze(1)).mean(-2)
    s2 = method("white_noise_2", Normal(-mu + additive, 1), n=1)
    # plus = lambda a, b: a + b
    # plus = storch.deterministic(plus)
    agg_v = agg_v + s2 * mu
    storch.add_cost(loss(agg_v), "loss")

storch.backward(debug=False, print_costs=True)
