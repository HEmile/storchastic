from torch.distributions import Distribution
from storch.tensor import Tensor
from storch.method import Method, Reparameterization
from storch.wrappers import _stochastic_parents
_COST_TENSORS = []


def sample(distr: Distribution, method: Method = None, n: int = 1) -> Tensor:
    if not method:
        if distr.has_rsample:
            method = Reparameterization()
    s_tensor = method.sample(distr, n)
    if _stochastic_parents:
        s_tensor._add_parents(_stochastic_parents)
    return s_tensor
