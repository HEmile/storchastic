from abc import ABC, abstractmethod
from torch.distributions import Distribution, Categorical, OneHotCategorical, Bernoulli, RelaxedOneHotCategorical, RelaxedBernoulli
from storch.tensor import DeterministicTensor, StochasticTensor
import torch
from typing import Optional
from storch.typing import DiscreteDistribution
from storch.util import has_differentiable_path
from pyro.distributions import RelaxedBernoulliStraightThrough, RelaxedOneHotCategoricalStraightThrough

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

class GumbelSoftmax(Method):
    """
    Method that automatically chooses between Gumbel Softmax relaxation and the score function depending on the
    differentiability requirements of cost nodes. Can only be used for reparameterizable distributions.
    Default option for reparameterizable distributions.
    """

    def __init__(self, straight_through=False):
        self.straight_through = straight_through
        self.temperature = 0.99 # TODO: How to design this parameter?

    def sample(self, distr: DiscreteDistribution, n: int) -> torch.Tensor:
        if isinstance(distr, (Categorical, OneHotCategorical)):
            if self.straight_through:
                gumbel_distr = RelaxedOneHotCategoricalStraightThrough(self.temperature, probs=distr.probs)
            else:
                gumbel_distr = RelaxedOneHotCategorical(self.temperature, probs=distr.probs)
        elif isinstance(distr, Bernoulli):
            if self.straight_through:
                gumbel_distr = RelaxedBernoulliStraightThrough(self.temperature, probs=distr.probs)
            else:
                gumbel_distr = RelaxedBernoulli(self.temperature, probs=distr.probs)
        else:
            raise ValueError("Using Gumbel Softmax with non-discrete distribution")
        return gumbel_distr.rsample((n,))


    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        if has_differentiable_path(cost_node, tensor):
            # There is a differentiable path, so we will just use reparameterization here.
            return None
        else:
            # No automatic baselines
            s = ScoreFunction()
            return s.estimator(tensor, cost_node, costs)


class ScoreFunction(Method):
    def sample(self, distr: Distribution, n: int) -> torch.Tensor:
        return distr.sample((n, ))

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> torch.Tensor:
        log_prob = tensor.distribution.log_prob(tensor._tensor)
        # Sum out over the even shape
        log_prob = log_prob.sum(dim=list(range(len(tensor.batch_links), len(log_prob.shape))))
        return log_prob * costs.detach()