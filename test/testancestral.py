import storch
import torch
import torch.distributions as td
method1 = storch.method.Reparameterization
method2 = storch.method.UnorderedSetEstimator

method2 = method2(plate_name="1",k=25)
p2 = td.Independent(td.OneHotCategorical(probs=torch.zeros([1000,3]).uniform_()),0)
samp2 = method2(p2)

# This hacky version works
method1 = method1(plate_name="2",n_samples=1)
hack = samp2[..., [0, 1]] + 1
p1 = td.Independent(td.Normal(loc=torch.zeros([1000,2]) * hack/ hack,scale=torch.ones([1000,2])),0)
samp1 = method1(p1)

# # But ideally we would like to automatically merge those plates together.
# method1 = method1(plate_name="1",n_samples=25)
# p1 = td.Independent(td.Normal(loc=torch.zeros([1000,2]),scale=torch.ones([1000,2])),0)
# samp1 = method1(p1)

print(samp1.shape) # torch.Size([25, 1000, 2])
print(samp2.shape) # torch.Size([25, 1000, 3])

print(storch.cat([samp1,samp2], -1).shape)  # torch.Size([25, 1000, 5])