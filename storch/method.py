from abc import ABC, abstractmethod
from torch.distributions import Distribution
from storch.tensor import DeterministicTensor, StochasticTensor
import torch
from typing import Optional
from storch.util import has_differentiable_path


class Method(ABC):
    @abstractmethod
    def sample(self, distr: Distribution, n: int) -> torch.Tensor:
        pass

    @abstractmethod
    # Estimators should optionally return a torch.Tensor that is going to be added to the total cost function
    # In the case of for example reparameterization, None can be returned to denote that no cost function is added
    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        pass


class Infer(Method):
    """
    Method that automatically chooses between reparameterization and the score function depending on the
    differentiability requirements of cost nodes. Can only be used for reparameterizable distributions.
    Default option for reparameterizable distributions.
    """
    def sample(self, distr: Distribution, n: int) -> StochasticTensor:
        if not distr.has_rsample:
            raise NotImplementedError("The input distribution has not implemented rsample. If you use a discrete "
                                      "distribution, make sure to use DiscreteReparameterization.")
        return distr.rsample((n,))

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        if has_differentiable_path(cost_node, tensor):
            # There is a differentiable path, so we will just use reparameterization here.
            return None
        else:
            # No automatic baselines
            s = ScoreFunction()
            return s.estimator(tensor, cost_node, costs)


class ScoreFunction(Method):
    def sample(self, distr: Distribution, n: int) -> StochasticTensor:
        return distr.sample((n, ))

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> torch.Tensor:
        log_prob = tensor.distribution.log_prob(tensor._tensor)
        # Sum out over the even shape
        log_prob = log_prob.sum(dim=list(range(len(tensor.batch_links), len(log_prob.shape))))
        return log_prob * costs.detach()