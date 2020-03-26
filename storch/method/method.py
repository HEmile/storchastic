from abc import ABC, abstractmethod
from torch.distributions import (
    Distribution,
    OneHotCategorical,
    Bernoulli,
)
from torch.nn import Parameter

from storch.tensor import CostTensor, StochasticTensor, Plate
import torch
import torch.nn.functional as F
from typing import Optional, Type, Union, Dict, Callable, List
from storch.util import (
    has_differentiable_path,
    get_distr_parameters,
    rsample_gumbel,
    split,
)
from storch.typing import DiscreteDistribution, BaselineFactory
import storch
from storch.method.baseline import MovingAverageBaseline, BatchAverageBaseline
import itertools
from torch.distributions.utils import clamp_probs


class Method(ABC, torch.nn.Module):
    @staticmethod
    def _create_hook(sample: StochasticTensor, name: str, plates: List[Plate]):
        accum_grads = sample.param_grads
        del sample  # Remove from hook closure for GC reasons

        def hook(grad: torch.Tensor):
            if name in accum_grads:
                accum_grads[name] = storch.Tensor(
                    accum_grads[name]._tensor + grad, [], plates, name + "_grad"
                )
            else:
                accum_grads[name] = storch.Tensor(grad, [], plates, name + "_grad")

        return hook

    def __init__(self):
        super().__init__()
        self._estimation_pairs = []
        self.register_buffer("iterations", torch.tensor(0, dtype=torch.long))

    def forward(
        self, sample_name: str, distr: Distribution, n: int = 1
    ) -> storch.tensor._StochasticTensorBase:
        return self.sample(sample_name, distr, n)

    def sample(
        self, sample_name: str, distr: Distribution, n: int = 1
    ) -> storch.tensor._StochasticTensorBase:
        # Unwrap the distributions parameters
        params: Dict[str, storch.Tensor] = get_distr_parameters(
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
                hook_plates = []
                if isinstance(param, storch.Tensor):
                    hook_plates = param.plates
                param._tensor.register_hook(
                    self._create_hook(s_tensor, name, hook_plates)
                )

        edited_sample = self.post_sample(s_tensor)
        if edited_sample is not None:
            new_s_tensor = storch.tensor._StochasticTensorBase(
                edited_sample._tensor,
                [s_tensor],
                s_tensor.plates,
                s_tensor.name,
                None,
                s_tensor.distribution,
                False,
                s_tensor.n,
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
        self, result_triples: [(StochasticTensor, CostTensor)]
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

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        return None


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
        return rsample_gumbel(distr, n, self.temperature, self.straight_through)

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


class Baseline(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(in_dim, 50)
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class LAX(Method):
    """
    Implements the LAX estimator from Grathwohl et al, 2018 https://arxiv.org/abs/1711.00123
    Code inspired by https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """

    def __init__(
        self, c_phi: Callable[[torch.Tensor], torch.Tensor] = None, in_dim=None
    ):
        super().__init__()
        if c_phi:
            self.c_phi = c_phi
        else:
            self.c_phi = Baseline(in_dim)
        # TODO: Add baseline strength

    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> (torch.Tensor, int):
        sample = distr.rsample((n,))
        return sample, n

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        return tensor.detach()

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        # Input rsampled value into c_phi
        output_baseline = self.c_phi(tensor)

        # Compute log probability. Make sure not to use the rsampled value: We want to compute the distribution
        # not through the sample but only through the distributional parameters.
        log_prob = tensor.distribution.log_prob(tensor.detach())
        log_prob = log_prob.sum(dim=list(range(tensor.plate_dims, len(log_prob.shape))))

        # Compute the derivative with respect to the distributional parameters through the baseline.
        derivs = []
        for param in get_distr_parameters(
            tensor.distribution, filter_requires_grad=True
        ).values():
            # param.register_hook(hook)
            if isinstance(param, storch.Tensor):
                param = param._tensor
            d_log_prob = torch.autograd.grad(
                [log_prob._tensor],
                [param],
                create_graph=True,
                grad_outputs=torch.ones_like(log_prob),
            )[0]
            d_output_baseline = torch.autograd.grad(
                [output_baseline._tensor],
                [param],
                create_graph=True,
                grad_outputs=torch.ones_like(output_baseline),
            )[0]
            derivs.append((param, d_log_prob, d_output_baseline))

        diff = cost_node - output_baseline
        var_loss = 0.0
        for param, d_log_prob, d_output_baseline in derivs:
            # Compute total derivative with respect to the parameter
            d_param = diff * d_log_prob + d_output_baseline
            # Reduce the plate of this sample in case multiple samples are taken
            d_param = storch.reduce_plates(d_param, plate_names=[tensor.name])
            # Compute backwards from the parameters using its total derivative
            param.backward(d_param._tensor, retain_graph=True)
            # Compute the gradient variance
            variance = (d_param ** 2).sum(d_param.event_dim_indices())
            var_loss += storch.reduce_plates(variance)

        c_phi_params = []

        for param in self.c_phi.parameters(recurse=True):
            if param.requires_grad:
                c_phi_params.append(param)
        d_variance = torch.autograd.grad([var_loss._tensor], c_phi_params)
        for i in range(len(c_phi_params)):
            c_phi_params[i].backward(d_variance[i])
        return None

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        return True


class RELAX(Method):
    """
    Implements the RELAX estimator from Grathwohl et al, 2018, https://arxiv.org/abs/1711.00123
    and the REBAR estimator from Tucker et al, 2017, https://arxiv.org/abs/1703.07370
    Code inspired by https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """

    def __init__(
        self,
        c_phi: Callable[[torch.Tensor], torch.Tensor] = None,
        in_dim=None,
        rebar=False,
    ):
        super().__init__()
        if c_phi:
            self.c_phi = c_phi
        else:
            self.c_phi = Baseline(in_dim)
        self.rebar = rebar
        self.temperature = Parameter(torch.tensor(1.0))
        # TODO: Automatically learn eta
        self.eta = 1.0

    def _sample_tensor(
        self, distr: Distribution, n: int, parents: [storch.Tensor], plates: [Plate]
    ) -> (torch.Tensor, int):
        relaxed_sample = rsample_gumbel(
            distr, n, self.temperature, straight_through=False
        )
        # In REBAR, the objective function is evaluated for the Gumbel sample, the conditional Gumbel sample \tilde{z} and the argmax of the Gumbel sample.
        if self.rebar:
            hard_sample = self._discretize(relaxed_sample, distr)
            cond_sample = self._conditional_rsample(hard_sample, distr)
            return torch.cat([relaxed_sample, hard_sample, cond_sample], 0), 3 * n
        else:
            return relaxed_sample, n
        # sample relaxed if not rebar else sample z then return -> (H(z), z, z|H(z) and n*3

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        if self.rebar:
            # Make sure the r-sampled gumbels don't backpropagate in the normal backward pass
            return tensor.detach()
        else:
            # Return H(z) for the function evaluation if using RELAX
            return self._discretize(tensor, tensor.distr)

    def plate_weighting(
        self, tensor: storch.StochasticTensor
    ) -> Optional[storch.Tensor]:
        # if REBAR: only weight over the true samples, put the relaxed samples to weight 0. This also makes sure
        # that they will not be backpropagated through in the cost backwards pass
        if self.rebar:
            n = tensor.n / 3
            weighting = tensor.new_zeros((3 * n,))
            weighting[:n] = tensor.new_tensor(1.0 / n)
            return weighting
        return super().plate_weighting(tensor)

    def _discretize(self, tensor: torch.Tensor, distr: Distribution) -> torch.Tensor:
        # Adapted from pyro.relaxed_straight_through
        if isinstance(distr, Bernoulli):
            return tensor.round()
        argmax = tensor.max(-1)[1]
        hard_sample = torch.zeros_like(tensor)
        if argmax.dim() < hard_sample.dim():
            argmax = argmax.unsqueeze(-1)
        return hard_sample.scatter_(-1, argmax, 1)

    def _conditional_rsample(
        self, hard_sample: torch.Tensor, distr: Distribution
    ) -> torch.Tensor:
        """
        Conditionally re-samples from the distribution given the hard sample.
        This samples z \sim p(z|b), where b is the hard sample and p(z) is a gumbel distribution.
        """
        # Adapted from torch.distributions.relaxed_bernoulli and torch.distributions.relaxed_categorical
        shape = hard_sample.shape
        probs = clamp_probs(distr.probs.expand(shape))
        v = clamp_probs(torch.rand(shape, dtype=probs.dtype, device=probs.device))
        if isinstance(distr, Bernoulli):
            pos_probs = probs[hard_sample == 1]
            v_prime = torch.zeros_like(hard_sample)
            # See https://arxiv.org/abs/1711.00123
            v_prime[hard_sample == 1] = v[hard_sample == 1] * pos_probs + (
                1 - pos_probs
            )
            v_prime[hard_sample == 0] = v[hard_sample == 0] * (
                1 - probs[hard_sample == 0]
            )
            log_sample = (
                probs.log() + probs.log1p() + v_prime.log() + v_prime.log1p()
            ) / self.temperature
            return log_sample.sigmoid()
        b = hard_sample.max(-1)[1]
        log_v = v.log()
        # See https://arxiv.org/abs/1711.00123
        # i != b (indexing could maybe be improved here, but i doubt it'd be more efficient)
        cond_gumbels = -(-(log_v / probs) - log_v).log()
        # i = b
        cond_gumbels[b] = -(-log_v).log()
        scores = cond_gumbels / self.temperature
        return (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        plate = tensor.get_plate(tensor.name)
        if self.rebar:
            hard_sample, relaxed_sample, cond_sample = tuple(
                split(tensor, plate, amt_slices=3)
            )
            hard_cost, relaxed_cost, cond_cost = tuple(
                split(cost_node, plate, amt_slices=3)
            )
        else:
            hard_sample = self._discretize(tensor, tensor.distribution)
            relaxed_sample = tensor
            cond_sample = self._conditional_rsample(tensor, tensor.distribution)

            hard_cost = cost_node
            relaxed_cost = 0.0
            cond_cost = 0.0

        # Input rsampled values into c_phi
        c_phi_relaxed = self.c_phi(relaxed_sample) + relaxed_cost
        c_phi_cond = self.c_phi(cond_sample) + cond_cost

        # Compute log probability of hard sample
        log_prob = tensor.distribution.log_prob(hard_sample)
        log_prob = log_prob.sum(
            dim=list(range(hard_sample.plate_dims, len(log_prob.shape)))
        )

        # Compute the derivative with respect to the distributional parameters through the baseline.
        derivs = []
        # TODO: It should either propagate over the logits or over the probs. Can we know which one is the parameter and
        # which one is computed dynamically?
        for param in get_distr_parameters(
            tensor.distribution, filter_requires_grad=True
        ).values():
            # param.register_hook(hook)
            # TODO: Can these be collected in a single torch.autograd.grad call?
            if isinstance(param, storch.Tensor):
                param = param._tensor
            d_log_prob = torch.autograd.grad(
                [log_prob._tensor],
                [param],
                create_graph=True,
                grad_outputs=torch.ones_like(log_prob),
            )[0]
            d_c_phi_relaxed = torch.autograd.grad(
                [c_phi_relaxed._tensor],
                [param],
                create_graph=True,
                grad_outputs=torch.ones_like(c_phi_relaxed),
            )[0]
            d_c_phi_cond = torch.autograd.grad(
                [c_phi_cond._tensor],
                [param],
                create_graph=True,
                grad_outputs=torch.ones_like(c_phi_cond),
            )[0]
            derivs.append((param, d_log_prob, d_c_phi_relaxed, d_c_phi_cond))

        diff = cost_node - self.eta * c_phi_cond
        var_loss = 0.0
        for param, d_log_prob, d_c_phi_relaxed, d_c_phi_cond in derivs:
            # Compute total derivative with respect to the parameter
            d_param = diff * d_log_prob + d_c_phi_relaxed - d_c_phi_cond
            # Reduce the plate of this sample in case multiple samples are taken
            d_param = storch.reduce_plates(d_param, plate_names=[tensor.name])
            # Compute backwards from the parameters using its total derivative
            param.backward(d_param._tensor, retain_graph=True)
            # Compute the gradient variance
            variance = (d_param ** 2).sum(d_param.event_dim_indices())
            var_loss += storch.reduce_plates(variance)

        c_phi_params = []

        for param in self.c_phi.parameters(recurse=True):
            if param.requires_grad:
                c_phi_params.append(param)
        d_variance = torch.autograd.grad([var_loss._tensor], c_phi_params)
        for i in range(len(c_phi_params)):
            c_phi_params[i].backward(d_variance[i])
        return None

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        return True
