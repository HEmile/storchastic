import torch.distributions as distr
import storch

_COST_TENSORS = []

def sample(distribution: distr.Distribution, method: storch.Method=None, N: int=1) -> storch.Tensor:
    return storch.Tensor.stochastic(distribution.rsample(), method)