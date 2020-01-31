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
    def estimator(self, tensor: StochasticTensor, costs: torch.Tensor, compute_statistics: bool) -> None:
        pass


class Reparameterization(Method):
    def sample(self, distr: Distribution, n: int) -> StochasticTensor:
        if not distr.has_rsample:
            raise NotImplementedError("The input distribution has not implemented rsample. If you use a discrete "
                                      "distribution, make sure to use DiscreteReparameterization.")
        s = distr.rsample((n,))
        return StochasticTensor(s, self, distr, s.requires_grad)

    def estimator(self, tensor: StochasticTensor, costs: torch.Tensor, compute_statistics: bool) -> None:
        if compute_statistics:
            grads = []
            params = get_distr_parameters(tensor.distribution)
            for cost in costs:
                grads.append(torch.autograd.grad(cost, params, retain_graph=True))
            tensor.grads = grads
