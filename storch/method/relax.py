import warnings

from functools import reduce
from operator import mul
from typing import Optional, Callable, Tuple

import torch
from torch.distributions import Distribution, Bernoulli

from torch.nn import Parameter

import storch
from storch import (
    Plate,
    CostTensor,
    StochasticTensor,
    deterministic,
    conditional_gumbel_rsample,
)
from storch.sampling import MonteCarlo, SamplingMethod
from storch.typing import Dims
from storch.method.method import Reparameterization, Method

import torch.nn.functional as F

from storch.util import get_distr_parameters, rsample_gumbel_softmax, split, magic_box


class Baseline(torch.nn.Module):
    def __init__(self, in_dim: Dims):
        super().__init__()
        self.reshape = False
        if not isinstance(in_dim, int):
            self.reshape = True
            in_dim = reduce(mul, in_dim)
        self.in_dim = in_dim
        self.fc1 = torch.nn.Linear(in_dim, 50)
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, x: storch.Tensor):
        if self.reshape or self.in_dim <= 1:
            x = x.reshape(x.shape[: x.plate_dims] + (-1,))
        return self.fc2(F.relu(self.fc1(x))).squeeze(-1)


class LAX(Reparameterization):
    """
    Gradient estimator for continuous random variables.
    Implements the LAX estimator from Grathwohl et al, 2018 https://arxiv.org/abs/1711.00123
    Code inspired by https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """

    def __init__(
        self,
        plate_name: str,
        *,
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
        c_phi: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        in_dim=None,
    ):
        """
        Either c_phi or in_dim needs to be non-None!
        :param plate_name: Name of the plate
        :param sampling_method: The sampling method to use
        :param n_samples: The amount of samples to take
        :param c_phi: The baseline network. Needs to be specified if `in_dim`  is not specified.
        :param in_dim: The size of the default baseline. Needs to be set if `c_phi` is not specified.
        """
        super().__init__(plate_name, sampling_method, n_samples)
        if c_phi:
            self.c_phi = c_phi
        else:
            self.c_phi = Baseline(in_dim)
        # TODO: Add baseline strength

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        """
        We do a reparameterized sample, but we have to make sure we detach afterwards. The rsample is to make sure we
        can backpropagate through the sample in the estimator.
        """
        return tensor.detach()

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Tuple[
        Optional[storch.Tensor], Optional[storch.Tensor], Optional[storch.Tensor]
    ]:
        # Input rsampled value into c_phi
        output_baseline = self.c_phi(tensor)

        # Compute log probability. Make sure not to use the rsampled value: We want to compute the distribution
        # not through the sample but only through the distributional parameters.
        log_prob = tensor.distribution.log_prob(tensor.detach())
        log_prob = log_prob.sum(dim=tensor.event_dim_indices)

        diff = cost_node - output_baseline  # [(...,) + (None,) * d_log_prob.event_dims]
        var_loss = 0.0
        # TODO: derivs is missing?
        for param, d_log_prob, d_output_baseline in derivs:
            # Compute total derivative with respect to the parameter
            d_param = diff * d_log_prob + d_output_baseline
            # Reduce the plate of this sample in case multiple samples are taken
            d_param = storch.reduce_plates(d_param, plates=[tensor.name])
            # Compute backwards from the parameters using its total derivative
            if isinstance(param, storch.Tensor):
                param = param._tensor
            param.backward(d_param._tensor, retain_graph=True)
            # Compute the gradient variance
            variance = (d_param ** 2).sum(d_param.event_dim_indices)
            var_loss += storch.reduce_plates(variance)

        c_phi_params = []

        for param in self.c_phi.parameters(recurse=True):
            if param.requires_grad:
                c_phi_params.append(param)
        d_variance = torch.autograd.grad([var_loss._tensor], c_phi_params)
        for i in range(len(c_phi_params)):
            c_phi_params[i].backward(d_variance[i])
        return log_prob

    def is_pathwise(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        return False

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor)]
    ) -> None:
        # During the normal backwards call, the parameters are accumulated gradients to the control variate parameters.
        # We don't want to minimize wrt to that loss, but to the one we define here.
        for param in self.control_params:
            param.grad = None
        tensors = []
        for tensor, _ in result_triples:
            if tensor not in tensors:
                tensors.append(tensor)
        # minimize the variance of the gradient with respect to the input parameters
        for tensor in tensors:
            d_param = next(iter(tensor.grad.values()))
            variance = (d_param ** 2).sum(d_param.event_dim_indices)
            var_loss = storch.reduce_plates(variance)

            d_variance = torch.autograd.grad(
                [var_loss._tensor], self.control_params, retain_graph=True,
            )
            print(d_variance)

            for i in range(len(self.control_params)):
                self.control_params[i].backward(d_variance[i])

    def should_create_higher_order_graph(self) -> bool:
        return True


def discretize(tensor: torch.Tensor, distr: Distribution) -> torch.Tensor:
    # Adapted from pyro.relaxed_straight_through
    if isinstance(distr, Bernoulli):
        return tensor.round()
    argmax = tensor.max(-1)[1]
    hard_sample = torch.zeros_like(tensor)
    if argmax.dim() < hard_sample.dim():
        argmax = argmax.unsqueeze(-1)
    return hard_sample.scatter_(-1, argmax, 1)


class RELAX(Method):
    """
    Gradient estimator for Bernoulli and Categorical distributions on any function.
    Implements the RELAX estimator from Grathwohl et al, 2018, https://arxiv.org/abs/1711.00123
    and the REBAR estimator from Tucker et al, 2017, https://arxiv.org/abs/1703.07370
    Code inspired by https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """

    def __init__(
        self,
        plate_name: str,
        *,
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
        c_phi: Callable[[torch.Tensor], torch.Tensor] = None,
        in_dim: Dims = None,
        rebar=False,
        train_eta=True,
    ):
        if not sampling_method:
            sampling_method = MonteCarlo(plate_name, n_samples)
        super().__init__(
            plate_name,
            sampling_method.set_mc_sample(self.sample_gumbel).set_mc_weighting_function(
                 self.plate_weighting
            ),
        )
        if c_phi:
            self.c_phi = c_phi
        elif in_dim:
            self.c_phi = Baseline(in_dim)
        else:
            raise ValueError("Either pass an explicit control variate c_phi or an input dimension")

        self.temperature = Parameter(torch.tensor(1.0))
        self.rebar = rebar
        self.eta = 1.0

        self.control_params = [self.temperature]
        if train_eta:
            self.eta = Parameter(torch.tensor(self.eta))
            self.control_params.append(self.eta)

        if isinstance(self.c_phi, torch.nn.Module):
            for c_phi_param in self.c_phi.parameters(recurse=True):
                if c_phi_param.requires_grad:
                    self.control_params.append(c_phi_param)

    def sample_gumbel(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        amt_samples: int,
    ):
        if self.rebar:
            relaxed_sample = rsample_gumbel_softmax(
                distr, amt_samples, self.temperature, straight_through=False
            )
            # TODO: For rebar, what if sample from a sequence, and amt_samples is 1? Then three are sampled (for each one, also hard_sample and cond_sample.)
            #    That'd still blow up... But I guess it's required?
            # In REBAR, the objective function is evaluated for the Gumbel sample, the conditional Gumbel sample \tilde{z} and the argmax of the Gumbel sample.
            hard_sample = discretize(relaxed_sample, distr)
            cond_sample = conditional_gumbel_rsample(
                hard_sample, distr.probs, isinstance(distr, torch.distributions.Bernoulli), self.temperature
            )

            # return (H(z), z, z|H(z)
            return torch.cat([hard_sample, relaxed_sample, cond_sample], 0)
        return rsample_gumbel_softmax(
            distr, amt_samples, self.temperature, straight_through=False
        )

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        if self.rebar:
            # Only weight over the true samples, put the relaxed samples to weight 0. This also makes sure
            # that they will not be backpropagated through in the cost backwards pass
            n = int(tensor.n // 3)
            weighting = tensor._tensor.new_zeros((tensor.n,))
            weighting[:n] = tensor._tensor.new_tensor(1.0 / n)
            return weighting
        return None

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        if self.rebar:
            # Make sure the r-sampled gumbels don't backpropagate in the normal backward pass
            # TODO: This doesn't work. We need to record the gradients with respect to the cost, as they are used in
            # the estimator. By detaching it, they are no longer recorded. However, not detaching means the normal loss
            # will capture it... Need to rethink this.
            return tensor
        else:
            # Return H(z) for the function evaluation if using RELAX
            return discretize(tensor, tensor.distribution)

    def compute_estimator(
        self, tensor: StochasticTensor, cost: CostTensor, plate_index: int, n: int, distribution: Distribution
    ) -> Tuple[storch.Tensor, storch.Tensor, storch.Tensor]:
        if self.rebar:
            log_prob = distribution.log_prob(tensor)
            @storch.deterministic
            def compute_REBAR_cvs(estimator: RELAX, tensor: StochasticTensor, cost: CostTensor):
                # TODO: Does it make more sense to implement REBAR by adding another plate dimension
                #  that controls the three different types of samples? (hard, relaxed, cond)?
                #  Then this can be implemented by simply reducing that plate, ie not requiring the weird zeros
                # Split the sampled tensor
                empty_slices = (slice(None),) * plate_index
                _index1 = empty_slices + (slice(n),)
                _index2 = empty_slices + (slice(n, 2 * n),)
                _index3 = empty_slices + (slice(2 * n, 3 * n),)
                relaxed_sample, cond_sample = (
                        tensor[_index2],
                        tensor[_index3],
                    )
                relaxed_cost, cond_cost = cost[_index2], cost[_index3]

                # Compute the control variates and log probabilities for the samples
                _c_phi_relaxed = estimator.c_phi(relaxed_sample) + relaxed_cost
                _c_phi_cond = estimator.c_phi(cond_sample) + cond_cost

                # Add zeros to ensure plates align
                c_phi_relaxed = torch.zeros_like(cost)
                c_phi_relaxed[_index1] = _c_phi_relaxed
                c_phi_cond = torch.zeros_like(cost)
                c_phi_cond[_index1] = _c_phi_cond
                return c_phi_relaxed, c_phi_cond

            return (log_prob,) + compute_REBAR_cvs(self, tensor, cost)
        else:
            hard_sample = discretize(tensor, distribution)
            relaxed_sample = tensor
            cond_sample = storch.conditional_gumbel_rsample(
                hard_sample, distribution.probs, isinstance(distribution, torch.distributions.Bernoulli), self.temperature
            )
            # Input rsampled values into c_phi
            _c_phi_relaxed = self.c_phi(relaxed_sample)
            _c_phi_cond = self.c_phi(cond_sample)
            _log_prob = distribution.log_prob(hard_sample)
            return _log_prob, _c_phi_relaxed, _c_phi_cond

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Tuple[
        Optional[storch.Tensor], Optional[storch.Tensor]
    ]:
        # TODO: This estimator likely doesn't reduce plate weightings properly

        plate = tensor.get_plate(tensor.name)
        plate_index = -1
        if plate.n > 1:
            plate_index = tensor.get_plate_dim_index(plate.name)

        log_prob, c_phi_z, c_phi_tilde_z = self.compute_estimator(
            tensor,
            cost_node,
            plate_index,
            plate.n // 3 if self.rebar else plate.n,
            tensor.distribution,
        )
        c_phi_z = self.eta * c_phi_z
        c_phi_tilde_z = self.eta * c_phi_tilde_z
        log_prob = log_prob.sum(log_prob.event_dim_indices)
        return (
            log_prob,
            c_phi_z - c_phi_z.detach()
            - c_phi_tilde_z + (2-magic_box(log_prob)) * c_phi_tilde_z.detach()
        )

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor)]
    ) -> None:
        # During the normal backwards call, the parameters are accumulated gradients to the control variate parameters.
        # We don't want to minimize wrt to that loss, but to the one we define here.
        for param in self.control_params:
            param.grad = None
        # Find grad of the distribution, then compute variance.
        # TODO: check if the set statement properly filters different results here
        tensors = list(set([result[0] for result in result_triples]))
        # minimize the variance of the gradient with respect to the input parameters
        for tensor in tensors:
            # TODO: We have to select the probs of the distribution here as that's what it flows to. Is this always correct?
            d_param = tensor.grad['probs']
            if not d_param.requires_grad:
                warnings.warn("Gradient of input tensor does not require grad which is needed to train the REBAR variance parameter.".format(tensor))
                continue
            variance = (d_param ** 2).sum(d_param.event_dim_indices)
            var_loss = storch.reduce_plates(variance)

            d_variance = torch.autograd.grad(
                [var_loss._tensor], self.control_params, retain_graph=True, allow_unused=True
            )
            for i in range(len(self.control_params)):
                self.control_params[i].backward(d_variance[i])

    def should_create_higher_order_graph(self) -> bool:
        return True


class REBAR(RELAX):
    """
    Gradient estimator for Bernoulli and Categorical distributions on differentiable functions.
    Implements REBAR as a special case of RELAX.
    See Tucker et al, 2017, https://arxiv.org/abs/1703.07370
    """

    def __init__(
        self,
        plate_name: str,
        *,
        c_phi: Callable[[torch.Tensor], torch.Tensor] = None,
        n_samples: int = 1,
    ):
        # Default REBAR does not use external control variate neural network c_phi
        if not c_phi:
            c_phi = lambda x: 0.0
        super().__init__(plate_name, n_samples=n_samples, c_phi=c_phi, rebar=True)
