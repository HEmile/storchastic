import torch
a = torch.tensor([[[[1., .4,.6, .9], [5., 4., 3., .2]]]])
b = torch.tensor([[[[True, False, True, False], [True, False, True, False]]]])

import storch
print(torch.Tensor.__getitem__(a, b))
print(a[b])
a[b] = 10.
print(a)
from torch.distributions import Normal, Categorical, Poisson
from storch.exceptions import IllegalStorchExposeError

method = storch.method.Infer(Poisson)

d = Normal(0, 1)

s = method.sample("oeps", d)
# print(s>1)
try:
    if s:
        print("Oeps")
except IllegalStorchExposeError:
    print("Good!!")

if s == s:
    print("ahh good")

d = Poisson(4)
s = method.sample(d)

print(s)

i = 0
for _ in range(s):
    i += 1