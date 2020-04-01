import storch
import torch
from torch.distributions import Bernoulli, OneHotCategorical

expect = storch.Expect()
probs = torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001, 0.995], requires_grad=True)
indices = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = OneHotCategorical(probs=probs)
z = expect.sample("z", b)
c = (2.4 * z * indices).sum(-1)
storch.add_cost(c, "no_baseline_cost")

storch.backward()

expect_grad = z.grad["probs"].clone()

method = storch.ScoreFunction(baseline_factory="batch_average")
grads = []
for i in range(10000):
    b = OneHotCategorical(probs=probs)
    z = method.sample("z", b, n=3)
    c = (2.4 * z * indices).sum(-1) + 100
    storch.add_cost(c, "baseline_cost")

    storch.backward()
    grad = z.grad["probs"].clone()
    grads.append(grad)
grad_samples = storch.gather_samples(grads, "variance")
mean = storch.reduce_plates(grad_samples, plate_names=["variance"])
print("mean grad", mean)
print("expected grad", expect_grad)
print("specific_diffs", (mean - expect_grad) ** 2)
bias = (storch.reduce_plates((mean - expect_grad) ** 2)).sum()
print(bias)

# That works... Adding constants to the costs doesn't change the gradient in expectation.
