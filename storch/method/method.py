from abc import ABC, abstractmethod
from torch.distributions import Distribution, OneHotCategorical, Bernoulli, Categorical

from storch.tensor import CostTensor, StochasticTensor, Plate
import torch
from typing import Optional, Type, Union, Dict, List
from storch.util import (
    has_differentiable_path,
    get_distr_parameters,
    rsample_gumbel,
)
from storch.typing import DiscreteDistribution, BaselineFactory
import storch
from storch.method.baseline import MovingAverageBaseline, BatchAverageBaseline
import itertools


class Method(ABC, torch.nn.Module):
    @staticmethod
    def _create_hook(sample: StochasticTensor, name: str, plates: List[Plate]):
        accum_grads = sample.param_grads
        del sample  # Remove from hook closure for GC reasons

        def hook(*args):
            # For some reason, this args unpacking is required for compatbility with registring on a .grad_fn...?
            # TODO: I'm sure there could be something wrong here
            grad = args[-1]
            if isinstance(grad, tuple):
                grad = grad[0]

            # print(grad)
            if name in accum_grads:
                accum_grads[name] = storch.Tensor(
                    accum_grads[name]._tensor + grad, [], plates, name + "_grad"
                )
            else:
                accum_grads[name] = storch.Tensor(grad, [], plates, name + "_grad")

        return hook

    def __init__(self, plate_name):
        super().__init__()
        self._estimation_pairs = []
        self.register_buffer("iterations", torch.tensor(0, dtype=torch.long))
        self.plate_name = plate_name

    def forward(self, distr: Distribution) -> storch.tensor.StochasticTensor:
        return self.sample(distr)

    def sample(self, distr: Distribution) -> storch.tensor.StochasticTensor:
        # Unwrap the distributions parameters
        params: Dict[str, storch.Tensor] = get_distr_parameters(
            distr, filter_requires_grad=False
        )
        parents: [torch.Tensor] = storch.wrappers._stochastic_parents.copy()
        # If we are in an @stochastic context, external plates might exist.
        plates: [Plate] = storch.wrappers._plate_links.copy()
        requires_grad = False
        for name, p in params.items():
            requires_grad = requires_grad or p.requires_grad
            if isinstance(p, storch.Tensor):
                parent_found = False
                for _p in parents:
                    if _p is p:
                        parent_found = True
                        break
                if not parent_found:
                    parents.append(p)
                # The sample should have the batch links of the parameters in the distribution, if present.
                for plate in p.plates:
                    if plate not in plates:
                        plates.append(plate)

        for plate in plates:
            if plate.name == self.plate_name:
                self.on_plate_already_present(plate)

        s_tensor, plate = self._sample_tensor(distr, parents, plates, requires_grad)

        batch_weighting = self.plate_weighting(s_tensor, plate)
        if batch_weighting is not None:
            plate.weight = batch_weighting
            # TODO: I don't think this code should be here.
            # if isinstance(batch_weighting, storch.Tensor):
            #     batch_weighting.plates[0] = plate

        for name, param in params.items():
            # TODO: Possibly could find the wrong gradients here if multiple distributions use the same parameter?
            # This maybe requires copying the tensor hm...
            if param.requires_grad:
                hook_plates = []
                if isinstance(distr, OneHotCategorical) or isinstance(
                    distr, Categorical
                ):
                    if param is not distr._param:
                        continue
                    # We only care about the input parameter. Ie, it returns both probs and logits, but only
                    # the one the user used to create the Distribution is of interest.
                    # These distributions first normalize their logits/probs. This causes incorrect gradient statistics.
                    # By taking a step back in the computation graph, we retrieve the correct parameter.
                    if isinstance(param, storch.Tensor):
                        hook_plates = param.plates
                        param = param._tensor
                    to_hook = param.grad_fn.next_functions[0][0]
                elif isinstance(param, storch.Tensor):
                    hook_plates = param.plates
                    to_hook = param._tensor
                else:
                    to_hook = param
                to_hook.register_hook(self._create_hook(s_tensor, name, hook_plates))

        # Possibly change something in the tensor now that it is wrapped and registered in the graph.
        # Used for example to rsample in LAX, and in post_sample detaching the tensor so that it won't record gradients
        # in the normal forward pass.
        edited_sample = self.post_sample(s_tensor)

        # Register this sampling method as being used in this iteration so that we can reset this method after the iteration
        if self not in storch.inference._sampling_methods:
            storch.inference._sampling_methods.append(self)

        if edited_sample is not None:
            new_s_tensor = storch.tensor.StochasticTensor(
                edited_sample._tensor,
                [s_tensor],
                s_tensor.plates,
                s_tensor.name,
                s_tensor.n,
                None,
                s_tensor.distribution,
                False,
            )
            new_s_tensor.param_grads = s_tensor.param_grads
            return new_s_tensor
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
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        requires_grad: bool,
    ) -> (storch.StochasticTensor, Plate):
        pass

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        # Estimators should optionally return a torch.Tensor that is going to be added to the total cost function.
        # In the case of for example reparameterization, None is returned to denote that no cost function is added.
        # When adding a loss function, adds_loss should return True
        return None

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor)]
    ) -> None:
        pass

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: Plate
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

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        return None

    def reset(self):
        """
        This function gets called whenever storchastic is reset. This is after storch.backward() or storch.reset() is
        called. Can be used to reset this methods state to some initial state that has to happen every iteration. 
        """
        pass

    def on_plate_already_present(self, plate: Plate):
        raise ValueError(
            "Cannot create stochastic tensor with name "
            + plate.name
            + ". A parent sample has already used this name. Use a different name for this sample."
        )


# For monte carlo sampled methods
class MonteCarloMethod(Method):
    """
    Monte Carlo methods use simple sampling methods that take n independent samples.
    Unlike complex ancestral sampling methods such as SampleWithoutReplacementMethod, the sampling behaviour is not dependent
    on earlier samples in the stochastic computation graph (but the distributions are!).
    """

    def __init__(self, plate_name: str, n_samples: int = 1):
        super().__init__(plate_name)
        self.n_samples = n_samples

    def _sample_tensor(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        requires_grad: bool,
    ) -> (storch.StochasticTensor, Plate):
        tensor = self.mc_sample(distr, parents, plates)
        plate_size = tensor.shape[0]
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)

        plate = Plate(self.plate_name, plate_size, plates.copy())
        plates.insert(0, plate)

        if isinstance(tensor, storch.Tensor):
            tensor = tensor._tensor

        s_tensor = StochasticTensor(
            tensor,
            parents,
            plates,
            self.plate_name,
            plate_size,
            self,
            distr,
            requires_grad,
        )
        return s_tensor, plate

    @abstractmethod
    def mc_sample(self, distr: Distribution, parents: [storch.Tensor], plates: [Plate]):
        pass


class Infer(MonteCarloMethod):
    """
    Method that automatically chooses reparameterization if it is available, otherwise the score function.
    """

    def __init__(
        self, plate_name: str, distribution_type: Type[Distribution], n_samples: int = 1
    ):
        super().__init__(plate_name, n_samples)
        if distribution_type.has_rsample:
            self._method = Reparameterization(plate_name, n_samples)
        elif issubclass(distribution_type, OneHotCategorical) or issubclass(
            distribution_type, Bernoulli
        ):
            self._method = GumbelSoftmax(plate_name, n_samples)
        else:
            self._method = ScoreFunction(plate_name, n_samples)
        self._score_method = ScoreFunction(plate_name, n_samples)

    def mc_sample(
        self, distr: Distribution, parents: [storch.Tensor], plates: [Plate]
    ) -> torch.Tensor:
        return self._method.mc_sample(distr, parents, plates)

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


class Reparameterization(MonteCarloMethod):
    """
    Can only be used for reparameterizable distributions.
    Default option for reparameterizable distributions.
    """

    def mc_sample(
        self, distr: Distribution, parents: [storch.Tensor], plates: [Plate]
    ) -> torch.Tensor:
        if not distr.has_rsample:
            raise ValueError(
                "The input distribution has not implemented rsample. If you use a discrete "
                "distribution, make sure to use eg GumbelSoftmax."
            )
        with storch.ignore_wrapping():
            return distr.rsample((self.n_samples,))

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
        plate_name: str,
        n_samples: int = 1,
        straight_through=False,
        initial_temperature=1.0,
        min_temperature=1.0e-4,
        annealing_rate=1.0e-5,
    ):
        super().__init__(plate_name, n_samples)
        self.straight_through = straight_through
        self.register_buffer("temperature", torch.tensor(initial_temperature))
        self.register_buffer("annealing_rate", torch.tensor(annealing_rate))
        self.register_buffer("min_temperature", torch.tensor(min_temperature))

    def mc_sample(
        self, distr: DiscreteDistribution, parents: [storch.Tensor], plates: [Plate],
    ) -> torch.Tensor:
        with storch.ignore_wrapping():
            return rsample_gumbel(
                distr, self.n_samples, self.temperature, self.straight_through
            )

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor, torch.Tensor)]
    ):
        if self.training:
            self.temperature = torch.max(
                self.min_temperature, torch.exp(-self.annealing_rate * self.iterations)
            )


class ScoreFunction(MonteCarloMethod):
    def __init__(
        self,
        plate_name: str,
        n_samples: int = 1,
        baseline_factory: Optional[Union[BaselineFactory, str]] = "moving_average",
        **kwargs,
    ):
        super().__init__(plate_name, n_samples)
        self.baseline_factory: Optional[BaselineFactory] = baseline_factory
        if isinstance(baseline_factory, str):
            if baseline_factory == "moving_average":
                # Baseline per cost possible? So this lookup/buffer thing is not necessary
                self.baseline_factory = lambda s, c: MovingAverageBaseline(**kwargs)
            elif baseline_factory == "batch_average":
                if n_samples == 1:
                    raise ValueError(
                        "Batch average can only be used for n_samples > 1. "
                    )
                self.baseline_factory = lambda s, c: BatchAverageBaseline()
            elif baseline_factory == "none" or baseline_factory == "None":
                self.baseline_factory = None
            else:
                raise ValueError("Invalid baseline name", baseline_factory)

    def mc_sample(
        self, distr: Distribution, parents: [storch.Tensor], plates: [Plate]
    ) -> torch.Tensor:
        with storch.ignore_wrapping():
            return distr.sample((self.n_samples,))

    def estimator(self, tensor: StochasticTensor, cost: CostTensor) -> storch.Tensor:
        log_prob = tensor.distribution.log_prob(tensor)
        if len(log_prob.shape) > tensor.plate_dims:
            # Sum out over the event shape
            log_prob = log_prob.sum(
                dim=list(range(tensor.plate_dims, len(log_prob.shape)))
            )

        if self.baseline_factory:
            baseline_name = "_b_" + tensor.name + "_" + cost.name
            if not hasattr(self, baseline_name):
                setattr(self, baseline_name, self.baseline_factory(tensor, cost))
            baseline = getattr(self, baseline_name)
            cost = cost - baseline.compute_baseline(tensor, cost)
        # print(cost)
        return log_prob * cost.detach()

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        return True


class Expect(Method):
    def __init__(self, plate_name: str, budget=10000):
        super().__init__(plate_name)
        self.budget = budget

    def _sample_tensor(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        requires_grad: bool,
    ) -> (storch.StochasticTensor, Plate):
        # TODO: Currently very inefficient as it isn't batched
        # TODO: What if the expectation has a parent
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
        cross_products = 1 if not sizes else None
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

        enumerate_tensor = enumerate_tensor.detach()

        plate_size = enumerate_tensor.shape[0]

        plate = Plate(self.plate_name, plate_size, plates.copy())
        plates.insert(0, plate)

        s_tensor = StochasticTensor(
            enumerate_tensor,
            parents,
            plates,
            self.plate_name,
            plate_size,
            self,
            distr,
            requires_grad,
        )
        return s_tensor, plate

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        # Weight by the probability of each possible event
        log_probs = tensor.distribution.log_prob(tensor)
        if log_probs.plate_dims < len(log_probs.shape):
            log_probs = log_probs.sum(
                dim=list(range(tensor.plate_dims, len(log_probs.shape)))
            )
        return log_probs.exp()
