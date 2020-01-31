from abc import ABC, abstractmethod
from torch.distributions import Distribution
from storch.tensor import StochasticTensor
import torch
from storch.util import get_distr_parameters


class Method(ABC):
    @abstractmethod
    def sample(self, distr: Distribution, n: int) -> StochasticTensor:
        pass

    @abstractmethod
    def estimator(self, tensor: StochasticTensor, costs: torch.Tensor, compute_statistics: bool) -> torch.Tensor:
        pass


class Reparameterization(Method):
    def sample(self, distr: Distribution, n: int) -> StochasticTensor:
        if not distr.has_rsample:
            raise NotImplementedError("The input distribution has not implemented rsample. If you use a discrete "
                                      "distribution, make sure to use DiscreteReparameterization.")
        s = distr.rsample((n,))
        return StochasticTensor(s, self, distr, s.requires_grad)

    def estimator(self, tensor: StochasticTensor, costs: torch.Tensor, compute_statistics: bool) -> torch.Tensor:
        if compute_statistics:
            grads = []
            params = get_distr_parameters(tensor.distribution, filter_requires_grad=True)
            # TODO: Requires looping over all n evaluations of the costs
            grads.append(torch.autograd.grad(costs, params, retain_graph=True))
            tensor.grads = grads
        return 1.


class ScoreFunction(Method):
    def sample(self, distr: Distribution, n: int) -> StochasticTensor:
        params = get_distr_parameters(distr, filter_requires_grad=True)
        s = distr.sample((n, ))
        return StochasticTensor(s, self, distr, len(params) > 0)

    def estimator(self, tensor: StochasticTensor, costs: torch.Tensor, compute_statistics: bool) -> torch.Tensor:
        log_prob = tensor.distribution.log_prob(tensor._tensor)
        return costs*log_prob # Yeah this won't work. It needs to return a multiplicative factor that the algorithm itself uses