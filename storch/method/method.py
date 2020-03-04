from abc import ABC, abstractmethod
from torch.distributions import Distribution, Categorical, OneHotCategorical, Bernoulli, RelaxedOneHotCategorical, RelaxedBernoulli
from storch.tensor import DeterministicTensor, StochasticTensor
import torch
from typing import Optional, Type, Union, Dict
from storch.util import has_differentiable_path, get_distr_parameters
from pyro.distributions import RelaxedBernoulliStraightThrough, RelaxedOneHotCategoricalStraightThrough
from storch.typing import DiscreteDistribution, BaselineFactory, Plate
from functools import reduce
from operator import mul
import storch
from storch.method.baseline import MovingAverageBaseline, BatchAverageBaseline

class Method(ABC, torch.nn.Module):
    @staticmethod
    def _create_hook(sample: StochasticTensor, tensor: torch.tensor, name: str):
        event_shape = list(tensor.shape)
        if len(sample.batch_shape) > 0:
            normalize_factor = 1. / reduce(mul, sample.batch_shape)
        else:
            normalize_factor = 1.

        accum_grads = sample._accum_grads
        del sample, tensor # For GC reasons

        def hook(grad: torch.Tensor):
            if not storch.inference._accum_grad:
                accum_grads[name] = grad
                return
            if name not in accum_grads:
                add_n = [sample.n] if sample.n > 1 else []
                accum_grads[name] = grad.new_zeros(add_n + event_shape)
            indices = []
            for link in sample.batch_links:
                indices.append(storch.inference._backward_indices[link])
            indices = tuple(indices)
            offset_indices = 1 if sample.n > 1 else 0
            # Unnormalizes the gradient to make them easier to use for computing statistics.
            accum_grads[name][indices] += grad[indices[offset_indices:]] / normalize_factor

        return hook

    def __init__(self):
        super().__init__()
        self._estimation_triples = []
        self.register_buffer('iterations', torch.tensor(0, dtype=torch.long))

    def forward(self, sample_name: str, distr: Distribution, n: int = 1) -> StochasticTensor:
        return self.sample(sample_name, distr, n)

    def sample(self, sample_name: str, distr: Distribution, n: int = 1) -> StochasticTensor:
        # Unwrap the distributions parameters
        params: Dict[str, torch.Tensor] = get_distr_parameters(distr, filter_requires_grad=False)
        parents: Dict[str, torch.Tensor] = {}
        plates: [Plate] = storch.wrappers._plate_links.copy()
        for name, p in params.items():
            if isinstance(p, storch.Tensor):
                try:
                    parents[name] = p
                    for plate in p.batch_links:
                        if plate not in plates:
                            plates.append(plate)
                    setattr(distr, name, p._tensor)
                    params[name] = p._tensor
                except AttributeError as e:
                    if storch._debug:
                        print("Couldn't unwrap parameter", name, "on distribution", distr)

        tensor: torch.Tensor = self._sample_tensor(distr, n)

        if n == 1:
            tensor = tensor.squeeze(0)
        s_tensor = StochasticTensor(tensor, storch.wrappers._stochastic_parents + list(parents.values()), self, plates, distr, len(params) > 0,
                                    n, sample_name)
        for name, param in params.items():
            # TODO: Possibly could find the wrong gradients here if multiple distributions use the same parameter?
            # This maybe requires copying the tensor hm...
            if param.requires_grad:
                # TODO: This requires_grad check might go wrong if requires_grad is assigned in a different way (like here with len(params) > 0
                param.register_hook(self._create_hook(s_tensor, param, name))
            if name in parents and not isinstance(param, storch.Tensor):
                setattr(distr, name, parents[name])

        return s_tensor

    def _estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        # For docs: costs here is aligned with the StochasticTensor, that's why there's two different things.
        self._estimation_triples.append((tensor, cost_node, costs))
        return self.estimator(tensor, cost_node, costs)

    def _update_parameters(self):
        self.iterations += 1
        self.update_parameters(self._estimation_triples)
        self._estimation_triples = []

    @abstractmethod
    def _sample_tensor(self, distr: Distribution, n: int) -> torch.Tensor:
        pass

    @abstractmethod
    # Estimators should optionally return a torch.Tensor that is going to be added to the total cost function
    # In the case of for example reparameterization, None can be returned to denote that no cost function is added
    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        pass

    def update_parameters(self, result_triples: [(StochasticTensor, DeterministicTensor, torch.Tensor)]) -> None:
        pass


class Infer(Method):
    """
    Method that automatically chooses the standard best gradient estimator for a distribution type.
    """

    def __init__(self, distribution_type: Type[Distribution]):
        super().__init__()
        if distribution_type.has_rsample:
            self._method = Reparameterization()
        elif issubclass(distribution_type, OneHotCategorical) or issubclass(distribution_type, Bernoulli):
            self._method = GumbelSoftmax()
        else:
            self._method = ScoreFunction()

    def _sample_tensor(self, distr: Distribution, n: int) -> torch.Tensor:
        return self._method._sample_tensor(distr, n)

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        return self._method.estimator(tensor, cost_node, costs)

    def update_parameters(self, result_triples: [(StochasticTensor, DeterministicTensor, torch.Tensor)]):
        self._method.update_parameters(result_triples)

class Reparameterization(Method):
    """
    Method that automatically chooses between reparameterization and the score function depending on the
    differentiability requirements of cost nodes. Can only be used for reparameterizable distributions.
    Default option for reparameterizable distributions.
    """

    def __init__(self):
        super().__init__()
        self._score_method = ScoreFunction()

    def _sample_tensor(self, distr: Distribution, n: int) -> StochasticTensor:
        if not distr.has_rsample:
            raise ValueError("The input distribution has not implemented rsample. If you use a discrete "
                                      "distribution, make sure to use eg GumbelSoftmax.")
        return distr.rsample((n,))

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        if has_differentiable_path(cost_node, tensor):
            # There is a differentiable path, so we will just use reparameterization here.
            return None
        else:
            # No automatic baselines
            return self._score_method.estimator(tensor, cost_node, costs)

    def update_parameters(self, result_triples: [(StochasticTensor, DeterministicTensor, torch.Tensor)]):
        self._score_method.update_parameters(result_triples)

class GumbelSoftmax(Method):
    """
    Method that automatically chooses between Gumbel Softmax relaxation and the score function depending on the
    differentiability requirements of cost nodes. Can only be used for reparameterizable distributions.
    Default option for reparameterizable distributions.
    """

    def __init__(self, straight_through=False, initial_temperature=1.0, min_temperature=1.e-4, annealing_rate=1.e-5):
        super().__init__()
        self.straight_through = straight_through
        self.register_buffer("temperature", torch.tensor(initial_temperature))
        self.register_buffer("annealing_rate", torch.tensor(annealing_rate))
        self.register_buffer("min_temperature", torch.tensor(min_temperature))

    def _sample_tensor(self, distr: DiscreteDistribution, n: int) -> torch.Tensor:
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
            return None
        else:
            s = ScoreFunction()
            return s.estimator(tensor, cost_node, costs)

    def update_parameters(self, result_triples: [(StochasticTensor, DeterministicTensor, torch.Tensor)]):
        if self.training:
            self.temperature = torch.max(self.min_temperature, torch.exp(-self.annealing_rate * self.iterations))


class ScoreFunction(Method):
    def __init__(self, baseline_factory: Optional[Union[BaselineFactory, str]] = "moving_average", **kwargs):
        super().__init__()
        self.baseline_factory: Optional[BaselineFactory] = baseline_factory
        if isinstance(baseline_factory, str):
            if baseline_factory == "moving_average":
                # Baseline per cost possible? So this lookup/buffer thing is not necessary
                self.baseline_factory = lambda s, c: MovingAverageBaseline(**kwargs)
            elif baseline_factory == "batch_average":
                self.baseline_factory = lambda s, c: BatchAverageBaseline()
            elif baseline_factory == "none" or baseline_factory == "None":
                self.baseline_factory = None
            else:
                raise ValueError("Invalid baseline name", baseline_factory)

    def _sample_tensor(self, distr: Distribution, n: int) -> torch.Tensor:
        return distr.sample((n, ))

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> torch.Tensor:
        log_prob = tensor.distribution.log_prob(tensor._tensor)
        # Sum out over the even shape
        log_prob = log_prob.sum(dim=list(range(len(tensor.batch_links), len(log_prob.shape))))

        if self.baseline_factory:
            baseline_name = "_b_" + tensor.name + "_" + cost_node.name
            if not hasattr(self, baseline_name):
                setattr(self, baseline_name, self.baseline_factory(tensor, cost_node))
            baseline = getattr(self, baseline_name)
            costs = costs - baseline.compute_baseline(tensor, cost_node, costs)
        return log_prob * costs.detach()


class Expect(Method):
    def __init__(self, budget=10000):
        super().__init__()
        self.budget = budget

    def _sample_tensor(self, distr: Distribution, n: int) -> torch.Tensor:
        if not distr.has_enumerate_support:
            raise ValueError("Can only calculate the expected value for distributions with enumerable support.")
        support = distr.enumerate_support(expand=True)
        # print(distr.batch_shape, distr.event_shape)
        # print(support[9, 1])

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        pass


class UnorderedSet(Method):
    def __init__(self):
        super().__init__()

    def _sample_tensor(self, distr: Distribution, n: int) -> torch.Tensor:
        pass

    def estimator(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> Optional[torch.Tensor]:
        pass