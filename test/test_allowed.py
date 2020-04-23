import torch

a = torch.tensor([[[[1.0, 0.4, 0.6, 0.9], [5.0, 4.0, 3.0, 0.2]]]])
b = torch.tensor([[[[True, False, True, False], [True, False, True, False]]]])

import storch

print(torch.Tensor.__getitem__(a, b))
print(a[b])
a[b] = 10.0
print(a)
from torch.distributions import Normal, Categorical, Poisson
from storch.exceptions import IllegalStorchExposeError

method = storch.method.Infer("oeps", Poisson)

d = Normal(0, 1)
s = method.sample(d)
# print(s>1)
try:
    if s:
        assert False
except IllegalStorchExposeError:
    pass

try:
    if s == s:
        assert False
except IllegalStorchExposeError:
    pass

d = Poisson(4)
s = method.sample(d).int()

print(s)

i = 0
try:
    for _ in range(s):
        i += 1
    assert False
except IllegalStorchExposeError:
    pass

try:
    if torch.allclose(s, s):
        assert False
except IllegalStorchExposeError:
    pass

try:
    if s.allclose(s):
        assert False
except IllegalStorchExposeError:
    pass
