from typing import Iterable

import storch
from torch.distributions import Normal, Categorical
import torch

torch.is_tensor = storch.is_tensor
torch.manual_seed(0)

mu_prior = torch.tensor([2.0, -3.0], requires_grad=True)
theta = torch.tensor([4.0, 5])

lax_method = storch.method.LAX("mu", in_dim=2)
expect = storch.method.Expect("k")
score_method = storch.method.ScoreFunction("white_noise_1", n_samples=2)
infer_method = storch.method.Infer("white_noise_2", Normal)


def loss(v):
    return torch.nn.MSELoss(reduction="none")(v, theta).mean(dim=-1)


mu = lax_method(Normal(mu_prior, 1))
k = expect(
    Categorical(
        probs=torch.tensor([[0.1, 0.3, 0.6], [0.1, 0.8, 0.1]], requires_grad=True)
    ),
)

agg_v = 0.0
s1 = 1.0
for i in range(2):
    k1, k2 = 0, 0
    if i == 1:
        k1 = k[:, 0]
        k2 = k[:, 1]
    s1 = score_method(Normal(mu + k1, 1))
    aaa = -mu + s1 * k2
    s2 = infer_method(Normal(-mu + s1 * k2, 1))
    # plus = lambda a, b: a + b
    # plus = storch.deterministic(plus)
    agg_v = agg_v + s1 + s2 * mu
    print(isinstance(agg_v, Iterable))
    storch.add_cost(loss(agg_v), "loss")

storch.backward(debug=False, print_costs=True)
