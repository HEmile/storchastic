from abc import ABC
from torch.distributions import Distribution, OneHotCategorical, Bernoulli, Categorical

from storch.tensor import CostTensor, StochasticTensor, Plate
import torch
from typing import Optional, Type, Union, Dict, List, Callable
from storch.util import (
    has_differentiable_path,
    get_distr_parameters,
    rsample_gumbel,
)
import storch
from storch.method.baseline import MovingAverageBaseline, BatchAverageBaseline, Baseline
from storch.sampling import (
    SamplingMethod,
    MonteCarlo,
    Enumerate,
)


class Method(ABC, torch.nn.Module):
    """
    Base class of gradient estimation methods.

    A :class:`Method` is a :class:`torch.nn.Module`, and can therefore contain parameters to optimize.
    Calling them (:meth:`forward`) with a PyTorch distribution returns a sampled Tensor of type :class:`storch.StochasticTensor`
    from that distribution that will use the gradient estimator in the backward pass when :func:`storch.backward` is called.

    Args:
        plate_name (str): The name of the :class:`.Plate` that samples of this method will use.
        sampling_method (storch.sampling.SamplingMethod): The method to sample tensors with given an input distribution.
    """

    def __init__(self, plate_name: str, sampling_method: SamplingMethod):
        super().__init__()
        self._estimation_pairs = []
        self.register_buffer("iterations", torch.tensor(0, dtype=torch.long))
        self.plate_name = plate_name
        self.sampling_method = sampling_method
        if not self.sampling_method.plate_name == plate_name:
            raise ValueError(
                "The plate name of the sampling method and the storch method should match."
            )

    def forward(self, distr: Distribution) -> storch.tensor.StochasticTensor:
        """
        Calls the sample method to sample from the given distribution
        :param torch.distribution.Distribution distr: The distribution to sample from.
        :return: The sampled tensor
        :rtype: storch.tensor.StochasticTensor
        """
        return self.sample(distr)

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

    def sample(self, distr: Distribution) -> storch.tensor.StochasticTensor:
        """
        Samples from the given distribution according to this Method's sampling scheme.
        :param distr:
        :return:
        """
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
                self.sampling_method.on_plate_already_present(plate)

        s_tensor: StochasticTensor
        s_tensor, plate = self.sampling_method(distr, parents, plates, requires_grad)

        s_tensor._set_method(self)

        batch_weighting = self.sampling_method.plate_weighting(s_tensor, plate)
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
                s_tensor.distribution,
                False,
            )
            new_s_tensor.param_grads = s_tensor.param_grads
            return new_s_tensor
        return s_tensor

    def _estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        """

        :param tensor:
        :param cost_node:
        :return:
        """
        self._estimation_pairs.append((tensor, cost_node))
        return self.estimator(tensor, cost_node)

    def _update_parameters(self):
        self.iterations += 1
        self.update_parameters(self._estimation_pairs)
        self._estimation_pairs = []

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        """
        Returns the surrogate loss of this estimator, if it adds one. A method such as :class:`storch.method.Reparameterization`
        will not add a surrogate loss.

        Note that this method is only called if :meth:`adds_loss` returns True.

        Args:
            tensor (storch.StochasticTensor): The sampled tensor to find the surrogate loss for.
            cost_node (storch.CostTensor):
        """
        return None

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor)]
    ) -> None:
        pass

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
        self.sampling_method.reset()


class Infer(Method):
    """
    Method that automatically chooses reparameterization if it is available, and otherwise the score function
    with appropriate baseline (moving average if n_samples = 1, else batch average).
    """

    def __init__(
        self,
        plate_name: str,
        distribution_type: Type[Distribution],
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
    ):
        if distribution_type.has_rsample:
            _method = Reparameterization(plate_name, sampling_method, n_samples)
        elif issubclass(distribution_type, OneHotCategorical) or issubclass(
            distribution_type, Bernoulli
        ):
            _method = GumbelSoftmax(plate_name, sampling_method, n_samples)
        else:
            _method = ScoreFunction(
                plate_name,
                sampling_method,
                n_samples,
                baseline_factory="moving_average"
                if n_samples == 1
                else "batch_average",
            )
        super().__init__(plate_name, _method.sampling_method)
        self._score_method = ScoreFunction(plate_name, sampling_method, n_samples)
        self._method = _method

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
    Can only be used for reparameterizable distributions and when the function to minimize is differentiable.
    Default option for reparameterizable distributions.
    """

    def __init__(
        self,
        plate_name: str,
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
    ):
        if not sampling_method:
            sampling_method = MonteCarlo(plate_name, n_samples)
        super().__init__(plate_name, sampling_method.set_mc_sample(self.reparam_sample))

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        if has_differentiable_path(cost_node, tensor):
            # There is a differentiable path, so we will just use reparameterization here.
            return False
        raise ValueError(
            "There is no differentiable path between the cost tensor and the stochastic tensor. "
            "We cannot use reparameterization. Use a different gradient estimator, or make sure your"
            "code is differentiable."
        )

    def reparam_sample(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        amt_samples: int,
    ):
        if not distr.has_rsample:
            raise ValueError(
                "The input distribution has not implemented rsample. If you use a discrete "
                "distribution, make sure to use eg GumbelSoftmax."
            )
        return distr.rsample((amt_samples,))


class GumbelSoftmax(Method):
    """
    Method that implements the Gumbel Softmax relaxation with temperature annealing.
    Can only be used to find the derivative when all paths to the cost nodes are differentiable.
    Introduced in https://arxiv.org/abs/1611.01144 and https://arxiv.org/abs/1611.00712
    """

    def __init__(
        self,
        plate_name: str,
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
        straight_through=False,
        initial_temperature=1.0,
        min_temperature=1.0e-4,
        annealing_rate=1.0e-5,
    ):
        if not sampling_method:
            sampling_method = MonteCarlo(plate_name, n_samples)
        super().__init__(
            plate_name, sampling_method.set_mc_sample(self.sample_gumbel),
        )

        self.straight_through = straight_through
        self.register_buffer("temperature", torch.tensor(initial_temperature))
        self.register_buffer("annealing_rate", torch.tensor(annealing_rate))
        self.register_buffer("min_temperature", torch.tensor(min_temperature))

    def sample_gumbel(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        amt_samples: int,
    ):
        return rsample_gumbel(
            distr, amt_samples, self.temperature, self.straight_through
        )

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        if has_differentiable_path(cost_node, tensor):
            # There is a differentiable path, so we will just use reparameterization here.
            return False
        raise ValueError(
            "There is no differentiable path between the cost tensor and the stochastic tensor. "
            "We cannot use reparameterization. Use a different gradient estimator, or make sure your"
            "code is differentiable."
        )


BaselineFactory = Callable[[storch.StochasticTensor, storch.CostTensor], Baseline]


class ScoreFunction(Method):
    """
    The score function multiplies the loss function with a log p(z) term to estimate the gradients. It is always
    applicable, but has high variance without variance reduction techniques.

    Args:
        plate_name (str): The name of the :class:`.Plate` that samples of this method will use.
        sampling_method (storch.sampling.SamplingMethod): The method to sample tensors with given an input distribution.
          if not given, this defaults to simple Monte Carlo sampling.
        n_samples (int): How many samples to use. This parameter is only used when :arg:`sampling_method` is not passed.
          Defaults to 1.
        baseline_factory: The :class:`storch.method.baseline.Baseline` to use. This is passed as a string (options:
          batch_average, moving_average) or as a BaselineFactory.
    """

    def __init__(
        self,
        plate_name: str,
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
        baseline_factory: Optional[Union[BaselineFactory, str]] = None,
        **kwargs,
    ):
        if not sampling_method:
            sampling_method = MonteCarlo(plate_name, n_samples)
        super().__init__(plate_name, sampling_method)
        self.baseline_factory: Optional[BaselineFactory] = baseline_factory
        if isinstance(baseline_factory, str):
            if baseline_factory == "moving_average":
                # Baseline per cost possible? So this lookup/buffer thing is not necessary
                self.baseline_factory = lambda s, c: MovingAverageBaseline(**kwargs)
            elif baseline_factory == "batch_average":
                if sampling_method.n_samples <= 1:
                    raise ValueError(
                        "Batch average can only be used for n_samples > 1. "
                    )
                self.baseline_factory = lambda s, c: BatchAverageBaseline()
            elif baseline_factory == "none" or baseline_factory == "None":
                self.baseline_factory = None
            else:
                raise ValueError("Invalid baseline name", baseline_factory)

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
        super().__init__(plate_name, Enumerate(plate_name, budget))
