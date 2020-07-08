from functools import reduce
from operator import mul
from typing import Optional, Callable

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
from storch.method.method import Reparameterization, GumbelSoftmax

import torch.nn.functional as F

from storch.util import get_distr_parameters, rsample_gumbel_softmax, split


class Baseline(torch.nn.Module):
    def __init__(self, in_dim: Dims):
        super().__init__()
        self.reshape = False
        if not isinstance(in_dim, int):
            self.reshape = True
            in_dim = reduce(mul, in_dim)

        self.fc1 = torch.nn.Linear(in_dim, 50)
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, x: storch.Tensor):
        if self.reshape:
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
    ) -> Optional[storch.Tensor]:
        # Input rsampled value into c_phi
        output_baseline = self.c_phi(tensor)

        # Compute log probability. Make sure not to use the rsampled value: We want to compute the distribution
        # not through the sample but only through the distributional parameters.
        log_prob = tensor.distribution.log_prob(tensor.detach())
        log_prob = log_prob.sum(dim=tensor.event_dim_indices)

        # Compute the derivative with respect to the distributional parameters through the baseline.
        derivs = []
        for param in get_distr_parameters(
            tensor.distribution, filter_requires_grad=True
        ).values():
            # param.register_hook(hook)
            d_log_prob = storch.grad(
                [log_prob],
                [param],
                create_graph=True,
                grad_outputs=torch.ones_like(log_prob),
            )[0]
            d_output_baseline = storch.grad(
                [output_baseline],
                [param],
                create_graph=True,
                grad_outputs=torch.ones_like(output_baseline),
            )[0]
            derivs.append((param, d_log_prob, d_output_baseline))

        diff = cost_node - output_baseline  # [(...,) + (None,) * d_log_prob.event_dims]
        var_loss = 0.0
        for param, d_log_prob, d_output_baseline in derivs:
            # Compute total derivative with respect to the parameter
            d_param = diff * d_log_prob + d_output_baseline
            # Reduce the plate of this sample in case multiple samples are taken
            d_param = storch.reduce_plates(d_param, plate_names=[tensor.name])
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
        return None

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
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


class RELAX(GumbelSoftmax):
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
    ):
        if not sampling_method:
            sampling_method = MonteCarlo(plate_name, n_samples)
        super().__init__(
            plate_name, sampling_method.set_mc_plate_weighting(self.plate_weighting),
        )
        if c_phi:
            self.c_phi = c_phi
        else:
            self.c_phi = Baseline(in_dim)

        self.temperature = Parameter(torch.tensor(1.0))
        self.rebar = rebar
        # TODO: Automatically learn eta
        self.eta = 1.0

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
                hard_sample, distr, self.temperature
            )

            # return (H(z), z, z|H(z)
            return torch.cat([hard_sample, relaxed_sample, cond_sample], 0)
        return super().sample_gumbel(distr, parents, plates, amt_samples)

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        if self.rebar:
            # Only weight over the true samples, put the relaxed samples to weight 0. This also makes sure
            # that they will not be backpropagated through in the cost backwards pass
            n = int(tensor.n / 3)
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
            return self._discretize(tensor, tensor.distribution)

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        # TODO: This estimator likely doesn't reduce plate weightings properly
        plate = tensor.get_plate(tensor.name)
        if self.rebar:
            hard_sample, relaxed_sample, cond_sample = split(
                tensor, plate, amt_slices=3
            )
            hard_cost, relaxed_cost, cond_cost = split(cost_node, plate, amt_slices=3)

        else:
            hard_sample = discretize(tensor, tensor.distribution)
            relaxed_sample = tensor
            cond_sample = storch.conditional_gumbel_rsample(
                hard_sample, tensor.distribution, self.temperature
            )

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
        param = tensor.distribution._param
        # TODO: It should either propagate over the logits or over the probs. Can we know which one is the parameter and
        # which one is computed dynamically?
        # param.register_hook(hook)
        # TODO: Can these be collected in a single torch.autograd.grad call?
        d_log_prob = storch.grad(
            [log_prob],
            [param],
            create_graph=True,
            grad_outputs=torch.ones_like(log_prob),
        )[0]
        d_c_phi_relaxed = storch.grad(
            [c_phi_relaxed],
            [param],
            create_graph=True,
            grad_outputs=torch.ones_like(c_phi_relaxed),
        )[0]
        d_c_phi_cond = storch.grad(
            [c_phi_cond],
            [param],
            create_graph=True,
            grad_outputs=torch.ones_like(c_phi_cond),
        )[0]

        diff = hard_cost - self.eta * c_phi_cond
        # Compute total derivative with respect to the parameter
        d_param = diff * d_log_prob + self.eta * (d_c_phi_relaxed - d_c_phi_cond)
        # Reduce the plate of this sample in case multiple samples are taken
        d_param = storch.reduce_plates(d_param, plates=[tensor.name])
        # Compute backwards from the parameters using its total derivative
        if isinstance(param, storch.Tensor):
            param._tensor.backward(d_param._tensor, retain_graph=True)
        else:
            param.backward(d_param._tensor, retain_graph=True)
        # Compute the gradient variance
        variance = (d_param ** 2).sum(d_param.event_dim_indices)
        var_loss = storch.reduce_plates(variance)

        # Minimize variance over the parameters of c_phi and the temperature (should it also minimize eta?)
        c_phi_params = [self.temperature]
        if isinstance(self.c_phi, torch.nn.Module):
            for c_phi_param in self.c_phi.parameters(recurse=True):
                if c_phi_param.requires_grad:
                    c_phi_params.append(c_phi_param)

        d_variance = torch.autograd.grad(
            [var_loss._tensor], c_phi_params, create_graph=self.rebar
        )

        for i in range(len(c_phi_params)):
            c_phi_params[i].backward(d_variance[i])
        return None

    def adds_loss(self, tensor: StochasticTensor, cost_node: CostTensor) -> bool:
        return True

    def update_parameters(
        self, result_triples: [(StochasticTensor, CostTensor)]
    ) -> None:
        pass


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
