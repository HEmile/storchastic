import storch
import torch
from torch.distributions import Bernoulli, OneHotCategorical

expect = storch.method.Expect("x")
probs = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.95], requires_grad=True)
indices = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = OneHotCategorical(probs=probs)
z = expect.sample(b)
c = (2.4 * z * indices).sum(-1)
storch.add_cost(c, "no_baseline_cost")

storch.backward()

expect_grad = z.grad["probs"].clone()

method = storch.method.UnorderedSetEstimator("x", k=6)
# method = storch.REBAR()
grads = []
for i in range(100):
    b = OneHotCategorical(probs=probs)
    z = method.sample(b)
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
mse = storch.reduce_plates((grad_samples - expect_grad) ** 2).sum()
print("MSE", mse)
bias = (storch.reduce_plates((mean - expect_grad) ** 2)).sum()
print("bias", bias)


# That works... Adding constants to the costs doesn't change the gradient in expectation.
