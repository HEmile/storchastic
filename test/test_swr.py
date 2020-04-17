import storch
from torch.distributions import Normal, OneHotCategorical
import torch

torch.manual_seed(0)

swr_method = storch.SampleWithoutReplacementMethod("z", 2)
normal_method1 = storch.Reparameterization("n1", 3)
normal_method2 = storch.Reparameterization("n2", 2)

l_entropy = torch.tensor([-3.0, -3.0, 2, -2.0], requires_grad=True)
h_entropy = torch.tensor([-0.1, 0.1, 0.05, -0.05], requires_grad=True)

n_params = torch.tensor(0.0, requires_grad=True)

# 7 independent samples from l_entropy
d1 = OneHotCategorical(logits=l_entropy.repeat((7, 1)))
d2 = OneHotCategorical(logits=h_entropy)

dn1 = Normal(n_params, 1.0)

z_1 = swr_method.sample(d1)
z_2 = swr_method.sample(d2)

print("z1", z_1)
print("z2", z_2)

n1 = normal_method1(dn1)
print("n1", n1)
_z_1 = z_1 + n1
_z_2 = z_1 + n1

print(_z_1, _z_2)

d3 = OneHotCategorical(logits=_z_1)
d4 = OneHotCategorical(logits=_z_2)

z_3 = swr_method.sample(d3)
z_4 = swr_method.sample(d4)

n2 = normal_method2.sample(dn1)
z_5 = swr_method.sample(d1)
_z_5 = z_5 + n2

print("z3", z_3)
print("z4", z_4)
print("z5", z_5)

d6 = OneHotCategorical(logits=z_3 + z_4 - _z_5)

z_6 = swr_method.sample(d6)

print("z6", z_6)
