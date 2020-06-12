import torch
from torch.distributions import Normal
from storch.method import Reparameterization, ScoreFunction
import storch

torch.manual_seed(0)


def compute_f(method):
    a = torch.tensor(5.0, requires_grad=True)
    b = torch.tensor(-3.0, requires_grad=True)
    c = torch.tensor(0.23, requires_grad=True)
    d = a + b

    # Sample e from a normal distribution using reparameterization
    normal_distribution = Normal(b + c, 1)
    e = method(normal_distribution)

    f = d * e * e
    return f, c


# e*e follows a noncentral chi-squared distribution https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
# exp_f = d * (1 + mu * mu)
repar = Reparameterization("e", n_samples=1)
f, c = compute_f(repar)
storch.add_cost(f, "f")
print(storch.backward())

print("first derivative estimate", c.grad)

f, c = compute_f(repar)
storch.add_cost(f, "f")
print(storch.backward())

print("second derivative estimate", c.grad)


def estimate_variance(method):
    gradient_samples = []
    for i in range(1000):
        f, c = compute_f(method)
        storch.add_cost(f, "f")
        storch.backward()
        gradient_samples.append(c.grad)
    gradients = storch.gather_samples(gradient_samples, "gradients")
    # print(gradients)
    print("variance", storch.variance(gradients, "gradients"))
    print("mean", storch.reduce_plates(gradients, "gradients"))
    print("st dev", torch.sqrt(storch.variance(gradients, "gradients")))

    print(type(gradients))
    print(gradients.shape)
    print(gradients.plates)


print("Reparameterization n=1")
estimate_variance(Reparameterization("e", n_samples=1))

print("Reparameterization n=10")
estimate_variance(Reparameterization("e", n_samples=10))

print("Score function n=1")
estimate_variance(ScoreFunction("e", n_samples=1))

print("Score function n=45")
estimate_variance(ScoreFunction("e", n_samples=45))

print("Score function with baseline n=20")
estimate_variance(ScoreFunction("e", n_samples=20, baseline_factory="batch_average"))
