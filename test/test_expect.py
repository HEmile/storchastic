import storch
import torch
from torch.distributions import Bernoulli, OneHotCategorical

expect = storch.Expect()
probs = torch.tensor([-1.0, -0.4, -40, 40, 2, 4, 0.1], requires_grad=True)
indices = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
b = OneHotCategorical(logits=probs)
z = expect.sample("z", b)
c = (2.4 * z * indices).sum(-1)
storch.add_cost(c, "no_baseline_cost")

storch.backward()

print(probs.grad)

probs.grad = None

b = OneHotCategorical(logits=probs)
z = expect.sample("z", b)
c = (2.4 * z * indices).sum(-1) + 100
storch.add_cost(c, "baseline_cost")

storch.backward()

print(probs.grad)


# That works... Adding constants to the costs doesn't change the gradient in expectation.
