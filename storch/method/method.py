from abc import ABC, abstractmethod
from torch.distributions import (
    Distribution,
    Categorical,
    OneHotCategorical,
    Bernoulli,
    RelaxedOneHotCategorical,
    RelaxedBernoulli,
)
from storch.tensor import CostTensor, StochasticTensor, Plate
import torch
from typing import Optional, Type, Union, Dict
from storch.util import has_differentiable_path, get_distr_parameters
from pyro.distributions import (
    RelaxedBernoulliStraightThrough,
    RelaxedOneHotCategoricalStraightThrough,
)
from storch.typing import DiscreteDistribution, BaselineFactory
import storch
from storch.method.baseline import MovingAverageBaseline, BatchAverageBaseline
import itertools


class Method(ABC, torch.nn.Module):
    @staticmethod
    def _create_hook(sample: StochasticTensor, name: str):
        accum_grads = sample.param_grads
        del sample  # Remove from hook closure for GC reasons

        def hook(grad: torch.Tensor):
            accum_grads[name] = grad

        return hook

    def __init__(self):
        super().__init__()
        self._estimation_pairs = []
        self.register_buffer("iterations", torch.tensor(0, dtype=torch.long))

    def forward(
        self, sample_name: str, distr: Distribution, n: int = 1
    ) -> StochasticTensor:
        return self.sample(sample_name, distr, n)

    def sample(
        self, sample_name: str, distr: Distribution, n: int = 1
    ) -> StochasticTensor:
        # Unwrap the distributions parameters
        params: Dict[str, torch.Tensor] = get_distr_parameters(
            distr, filter_requires_grad=False
        )
        parents: Dict[str, torch.Tensor] = {}
        # If we are in an @stochastic context, external plates might exist.
        plates: [Plate] = storch.wrappers._plate_links.copy()
        for name, p in params.items():
            if isinstance(p, storch.Tensor):
                parents[name] = p
                # The sample should have the batch links of the parameters in the distribution, if present.
                for plate in p.plates:
                    if plate not in plates:
                        plates.append(plate)
                params[name] = p._tensor

        # Will not rewrap in @deterministic, because sampling statements will insert an additional dimensions in the
        # first batch dimension, causing the rewrapping statement to error as it violates the plating constraints.
        storch.wrappers._ignore_wrap = True
        tensor, batch_size = self._sample_tensor(distr, n, parents, plates)
        storch.wrappers._ignore_wrap = False

        if batch_size == 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)

        s_tensor = StochasticTensor(
            tensor,
            storch.wrappers._stochastic_parents + list(parents.values()),
            self,
            plates,
            distr,
            len(params) > 0,
            batch_size,
            sample_name,
        )

        for name, param in params.items():
            # TODO: Possibly could find the wrong gradients here if multiple distributions use the same parameter?
            # This maybe requires copying the tensor hm...
            if param.requires_grad:
                param.register_hook(self._create_hook(s_tensor, name))

        return s_tensor

    def _estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        self._estimation_pairs.append((tensor, cost_node))
        return self.estimator(tensor, cost_node)

    def _update_parameters(self):
        self.iterations += 1
        self.update_parameters(self._estimation_pairs)
        self._estimation_pairs = []

    @abstractmethod
    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> (torch.Tensor, int):
        pass

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        # Estimators should optionally return a torch.Tensor that is going to be added to the total cost function.
        # In the case of for example reparameterization, None is returned to denote that no cost function is added.
        # When adding a loss function, adds_loss should return True
        return None

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor, torch.Tensor)]
    ) -> None:
        pass

    def plate_weighting(
        self, tensor: storch.StochasticTensor
    ) -> Optional[storch.Tensor]:
        # Weight by the size of the sample. Overload this if your gradient estimation uses some kind of weighting
        # of the different events, like importance sampling or computing the expectation.
        # If None is returned, it is assumed the samples are iid monte carlo samples.
        return None

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        """
        Returns true if the estimator will add an additional loss function for the derivative of the parameters
        of the stochastic tensor with respect to the cost node.
        """
        return False


class Infer(Method):
    """
    Method that automatically chooses reparameterization if it is available, otherwise the score function.
    """

    def __init__(self, distribution_type: Type[Distribution]):
        super().__init__()
        if distribution_type.has_rsample:
            self._method = Reparameterization()
        elif issubclass(distribution_type, OneHotCategorical) or issubclass(
            distribution_type, Bernoulli
        ):
            self._method = GumbelSoftmax()
        else:
            self._method = ScoreFunction()
        self._score_method = ScoreFunction()

    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> (torch.Tensor, int):
        return self._method._sample_tensor(distr, n, parents, plates)

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[torch.Tensor]:
        return self._score_method.estimator(tensor, cost_node)

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor, torch.Tensor)]
    ):
        self._method.update_parameters(result_triples)
        self._score_method.update_parameters(result_triples)

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        if has_differentiable_path(cost_node, tensor):
            # There is a differentiable path, so we will just use reparameterization here.
            return False
        else:
            # No automatic baselines. Use the score function.
            return True


class Reparameterization(Method):
    """
    Method that automatically chooses between reparameterization and the score function depending on the
    differentiability requirements of cost nodes. Can only be used for reparameterizable distributions.
    Default option for reparameterizable distributions.
    """

    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> (torch.Tensor, int):
        if not distr.has_rsample:
            raise ValueError(
                "The input distribution has not implemented rsample. If you use a discrete "
                "distribution, make sure to use eg GumbelSoftmax."
            )
        return distr.rsample((n,)), n

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        if has_differentiable_path(cost_node, tensor):
            # There is a differentiable path, so we will just use reparameterization here.
            return False
        raise ValueError(
            "There is no differentiable path between the cost tensor and the stochastic tensor. "
            "We cannot use reparameterization. Use a different gradient estimator, or make sure your"
            "code is differentiable."
        )


class GumbelSoftmax(Reparameterization):
    """
    Method that automatically chooses between Gumbel Softmax relaxation and the score function depending on the
    differentiability requirements of cost nodes. Can only be used for reparameterizable distributions.
    Default option for reparameterizable distributions.
    """

    def __init__(
        self,
        straight_through=False,
        initial_temperature=1.0,
        min_temperature=1.0e-4,
        annealing_rate=1.0e-5,
    ):
        super().__init__()
        self.straight_through = straight_through
        self.register_buffer("temperature", torch.tensor(initial_temperature))
        self.register_buffer("annealing_rate", torch.tensor(annealing_rate))
        self.register_buffer("min_temperature", torch.tensor(min_temperature))

    def _sample_tensor(
        self,
        distr: DiscreteDistribution,
        n: int,
        parents: [storch.Tensor],
        plates: [Plate],
    ) -> (torch.Tensor, int):
        if isinstance(distr, (Categorical, OneHotCategorical)):
            if self.straight_through:
                gumbel_distr = RelaxedOneHotCategoricalStraightThrough(
                    self.temperature, probs=distr.probs
                )
            else:
                gumbel_distr = RelaxedOneHotCategorical(
                    self.temperature, probs=distr.probs
                )
        elif isinstance(distr, Bernoulli):
            if self.straight_through:
                gumbel_distr = RelaxedBernoulliStraightThrough(
                    self.temperature, probs=distr.probs
                )
            else:
                gumbel_distr = RelaxedBernoulli(self.temperature, probs=distr.probs)
        else:
            raise ValueError("Using Gumbel Softmax with non-discrete distribution")
        return gumbel_distr.rsample((n,)), n

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor, torch.Tensor)]
    ):
        if self.training:
            self.temperature = torch.max(
                self.min_temperature, torch.exp(-self.annealing_rate * self.iterations)
            )


class ScoreFunction(Method):
    def __init__(
        self,
        baseline_factory: Optional[Union[BaselineFactory, str]] = "moving_average",
        **kwargs
    ):
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

    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> (torch.Tensor, int):
        return distr.sample((n,)), n

    def estimator(self, tensor: StochasticTensor, cost: CostTensor) -> storch.Tensor:
        log_prob = tensor.distribution.log_prob(tensor)
        # Sum out over the event shape
        log_prob = log_prob.sum(dim=list(range(tensor.plate_dims, len(log_prob.shape))))

        if self.baseline_factory:
            baseline_name = "_b_" + tensor.name + "_" + cost.name
            if not hasattr(self, baseline_name):
                setattr(self, baseline_name, self.baseline_factory(tensor, cost))
            baseline = getattr(self, baseline_name)
            cost = cost - baseline.compute_baseline(tensor, cost)
        return log_prob * cost.detach()

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        return True


class Expect(Method):
    def __init__(self, budget=10000):
        super().__init__()
        self.budget = budget

    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> (torch.Tensor, int):
        # TODO: Currently very inefficient as it isn't batched
        if not distr.has_enumerate_support:
            raise ValueError(
                "Can only calculate the expected value for distributions with enumerable support."
            )
        support: torch.Tensor = distr.enumerate_support(expand=True)
        support_non_expanded: torch.Tensor = distr.enumerate_support(expand=False)
        expect_size = support.shape[0]

        batch_len = len(plates)
        sizes = support.shape[
            batch_len + 1 : len(support.shape) - len(distr.event_shape)
        ]
        amt_samples_used = expect_size
        cross_products = None
        for dim in sizes:
            amt_samples_used = amt_samples_used ** dim
            if not cross_products:
                cross_products = dim
            else:
                cross_products = cross_products ** dim

        if amt_samples_used > self.budget:
            raise ValueError(
                "Computing the expectation on this distribution would exceed the computation budget."
            )

        enumerate_tensor = support.new_zeros(
            [amt_samples_used] + list(support.shape[1:])
        )
        support_non_expanded = support_non_expanded.squeeze().unsqueeze(1)
        for i, t in enumerate(
            itertools.product(support_non_expanded, repeat=cross_products)
        ):
            enumerate_tensor[i] = torch.cat(t, dim=0)
        return enumerate_tensor.detach(), enumerate_tensor.shape[0]

    def plate_weighting(
        self, tensor: storch.StochasticTensor
    ) -> Optional[storch.Tensor]:
        # Weight by the probability of each possible event
        log_probs = tensor.distribution.log_prob(tensor)
        return log_probs.sum(
            dim=list(range(tensor.plate_dims, len(log_probs.shape)))
        ).exp()


class UnorderedSet(Method):
    def __init__(self):
        super().__init__()

    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> torch.Tensor:
        pass

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        pass
