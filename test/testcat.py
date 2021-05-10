import storch
import torch
import torch.distributions as td
method1 = storch.method.Reparameterization
method2 = storch.method.ScoreFunction

method1 = method1(plate_name="1",n_samples=25)
method2 = method2(plate_name="1",n_samples=25)
p1 = td.Independent(td.Normal(loc=torch.zeros([1000, 2]), scale=torch.ones([1000, 2])), 0)
p2 = td.Independent(td.OneHotCategorical(probs=torch.zeros([1000, 3]).uniform_()), 0)

samp1 = method1(p1)
samp2 = method2 (p2)
# torch.Size([25, 1000, 2])
# torch.Size([25, 1000, 3])

print(storch.cat([samp1,samp2], 2).shape)
# torch.Size([25, 1000, 5])

method1 = storch.method.Reparameterization
method2 = storch.method.UnorderedSetEstimator

method1 = method1(plate_name="1",n_samples=25)
method2 = method2(plate_name="2",k=25)
p1 = td.Independent(td.Normal(loc=torch.zeros([1000, 2]), scale=torch.ones([1000, 2])), 0)
p2 = td.Independent(td.OneHotCategorical(probs=torch.zeros([1000, 3]).uniform_()), 0)

samp1 = method1(p1)
samp2 = method2 (p2)
# torch.Size([25, 1000, 2])
# torch.Size([25, 1000, 3])

print(storch.cat([samp1,samp2], -1).shape)
# torch.Size([25, 25, 1000, 5])