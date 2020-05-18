import storch
from torch.distributions import Normal, OneHotCategorical
import torch

torch.is_tensor = storch.is_tensor
torch.manual_seed(0)

# LEGEND OF SHAPES
# These are all different and > 1 to make them identifiable.
# ================
# k: Size of Score Function samples (of which 1 is the kappa dimension)
# |D_yv|: Size of sample from categorical (=4)
# event: event created by repeating in d1
# n1: Plate size of first normal
# n2: Plate size of second normal

k = 4
d_yv = 4
event = 2
plt_n1 = 3
plt_n2 = 2

# Define swr method
swr_method = storch.method.ScoreFunctionWOR("z", k, biased=True, use_baseline=False)
normal_method1 = storch.method.ScoreFunction("n1", n_samples=plt_n1)


l_entropy = torch.tensor([-3.0, -3.0, 2, -2.0], requires_grad=True)
h_entropy = torch.tensor([-0.1, 0.1, 0.05, -0.05], requires_grad=True)

n_params = torch.tensor(0.0, requires_grad=True)


d1 = OneHotCategorical(logits=l_entropy.repeat((event, 1)))
d2 = OneHotCategorical(logits=h_entropy)

dn1 = Normal(n_params, 1.0)

# k x event x |D_yv|
z_1 = swr_method.sample(d1)
# k x |D_yv|
z_2 = swr_method.sample(d2)

print("z1", z_1)
print("z2", z_2)

assert z_1.shape == (min(k, d_yv ** event), event, d_yv)
assert z_2.shape == (min(k, d_yv ** (event + 1)), d_yv)

# n1
n1 = normal_method1(dn1)
print("n1", n1)
_z_1 = z_1 + n1
_z_2 = z_2 + n1

print(_z_1, _z_2)

d3 = OneHotCategorical(logits=_z_1)
d4 = OneHotCategorical(logits=_z_2)

# k x n1 x event x |D_yv|
z_3 = swr_method.sample(d3)
# k x n1 x |D_yv|
z_4 = swr_method.sample(d4)

print("z3", z_3)
print("z4", z_4)

assert z_3.shape == (plt_n1, k, event, d_yv) or z_3.shape == (k, plt_n1, event, d_yv)
assert z_4.shape == (plt_n1, k, d_yv) or z_4.shape == (k, plt_n1, d_yv)

normal_method2 = storch.method.ScoreFunction("n2", n_samples=plt_n2)
# n2
n2 = normal_method2.sample(dn1)
# n2 x |D_yv|
n_l_entropy = l_entropy + n2.unsqueeze(-1)
d5 = OneHotCategorical(logits=n_l_entropy)

# This is a strange one: although n_l_entropy doesn't include n1, it should be included here, as the ancestral parents
# are dependent on n1!
# k x (n1 x n2 or n2 x n1) x |D_yv|
z_5 = swr_method.sample(d5)

print("z5", z_5)
if isinstance(swr_method.sampling_method, storch.sampling.SampleWithoutReplacement):
    assert z_5.shape == (plt_n1, plt_n2, k, d_yv) or z_5.shape == (
        plt_n2,
        plt_n1,
        k,
        d_yv,
    )
else:
    assert z_5.shape == (plt_n2, k, d_yv) or z_5.shape == (k, plt_n2, d_yv)

d6 = OneHotCategorical(logits=z_3 + z_4.unsqueeze(-2) - z_5.unsqueeze(-2))

z_6 = swr_method.sample(d6)

print("z6", z_6)

assert z_6.shape == (plt_n1, plt_n2, k, event, d_yv) or z_6.shape == (
    plt_n2,
    plt_n1,
    k,
    event,
    d_yv,
)
# Sum the amount of event 1's being true.
cost = torch.sum(z_6[..., 0], -1)
storch.add_cost(cost, "cost")
storch.backward()

# Print what values of z1 are selected in the final sample step. As it has very low entropy, this should be all [0,0,1,0]
# print("final z1", z_6.plates[2].on_unwrap_tensor(z_1))
