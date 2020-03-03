import storch
from torch.distributions import Normal, Categorical, Poisson
from storch.exceptions import IllegalConditionalError


method = storch.method.Infer(Poisson)

d = Normal(0, 1)

s = method.sample("oeps", d)
# print(s>1)
try:
    if s:
        print("Oeps")
except IllegalConditionalError:
    print("Good!!")

if s == s:
    print("ahh good")

d = Poisson(4)
s = method.sample(d)

print(s)

i = 0
for _ in range(s):
    i += 1