from functools import reduce
from operator import mul
from typing import Optional, Callable

import torch
from torch.distributions import Distribution, Bernoulli
from torch.distributions.utils import clamp_probs
from torch.nn import Parameter

import storch
from storch import Plate, CostTensor, StochasticTensor, deterministic
from storch.method.method import MonteCarloMethod
from storch.typing import Dims

import torch.nn.functional as F

from storch.util import get_distr_parameters, rsample_gumbel, split


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


class LAX(MonteCarloMethod):
    """
    Gradient estimator for continuous random variables.
    Implements the LAX estimator from Grathwohl et al, 2018 https://arxiv.org/abs/1711.00123
    Code inspired by https://github.com/duvenaud/relax/blob/master/pytorch_toy.py
    """

    def __init__(
        self,
        plate_name: str,
        *,
        n_samples: int = 1,
        c_phi: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        in_dim=None,
    ):
        super().__init__(plate_name, n_samples)
        if c_phi:
            self.c_phi = c_phi
        else:
            self.c_phi = Baseline(in_dim)
        # TODO: Add baseline strength

    def mc_sample(
        self, distr: Distribution, parents: [storch.Tensor], plates: [Plate]
    ) -> torch.Tensor:
        sample = distr.rsample((self.n_samples,))
        return sample

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
        log_prob = log_prob.sum(dim=tensor.event_dim_indices())

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


class RELAX(MonteCarloMethod):
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
        n_samples: int = 1,
        c_phi: Callable[[torch.Tensor], torch.Tensor] = None,
        in_dim: Dims = None,
        rebar=False,
    ):
        super().__init__(plate_name, n_samples)
        if c_phi:
            self.c_phi = c_phi
        else:
            self.c_phi = Baseline(in_dim)
        self.rebar = rebar
        self.temperature = Parameter(torch.tensor(1.0))
        # TODO: Automatically learn eta
        self.eta = 1.0

    def mc_sample(
        self, distr: Distribution, parents: [storch.Tensor], plates: [Plate]
    ) -> torch.Tensor:
        relaxed_sample = rsample_gumbel(
            distr, self.n_samples, self.temperature, straight_through=False
        )
        # In REBAR, the objective function is evaluated for the Gumbel sample, the conditional Gumbel sample \tilde{z} and the argmax of the Gumbel sample.
        if self.rebar:
            hard_sample = self._discretize(relaxed_sample, distr)
            cond_sample = self._conditional_gumbel_rsample(hard_sample, distr)
            return torch.cat([hard_sample, relaxed_sample, cond_sample], 0)
        else:
            return relaxed_sample
        # sample relaxed if not rebar else sample z then return -> (H(z), z, z|H(z) and n*3

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

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        # if REBAR: only weight over the true samples, put the relaxed samples to weight 0. This also makes sure
        # that they will not be backpropagated through in the cost backwards pass
        if self.rebar:
            n = int(tensor.n / 3)
            weighting = tensor.new_zeros((3 * n,))
            weighting[:n] = tensor.new_tensor(1.0 / n)
            return weighting
        return super().plate_weighting(tensor, plate)

    def _discretize(self, tensor: torch.Tensor, distr: Distribution) -> torch.Tensor:
        # Adapted from pyro.relaxed_straight_through
        if isinstance(distr, Bernoulli):
            return tensor.round()
        argmax = tensor.max(-1)[1]
        hard_sample = torch.zeros_like(tensor)
        if argmax.dim() < hard_sample.dim():
            argmax = argmax.unsqueeze(-1)
        return hard_sample.scatter_(-1, argmax, 1)

    @deterministic
    def _conditional_gumbel_rsample(
        self, hard_sample: torch.Tensor, distr: Distribution
    ) -> torch.Tensor:
        """
        Conditionally re-samples from the distribution given the hard sample.
        This samples z \sim p(z|b), where b is the hard sample and p(z) is a gumbel distribution.
        """
        # Adapted from torch.distributions.relaxed_bernoulli and torch.distributions.relaxed_categorical
        shape = hard_sample.shape
        probs = (
            distr.probs
            if not isinstance(hard_sample, storch.Tensor)
            else distr.probs._tensor
        )
        probs = clamp_probs(probs.expand_as(hard_sample))
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
        # b=argmax(hard_sample)
        b = hard_sample.max(-1).indices
        # b = F.one_hot(b, hard_sample.shape[-1])

        # See https://arxiv.org/abs/1711.00123
        log_v = v.log()
        # i != b (indexing could maybe be improved here, but i doubt it'd be more efficient)
        log_v_b = torch.gather(log_v, -1, b.unsqueeze(-1))
        cond_gumbels = -(-(log_v / probs) - log_v_b).log()
        # i = b
        index_sample = hard_sample.bool()
        cond_gumbels[index_sample] = -(-log_v[index_sample]).log()
        scores = cond_gumbels / self.temperature
        return (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()

    def estimator(
        self, tensor: StochasticTensor, cost_node: CostTensor
    ) -> Optional[storch.Tensor]:
        plate = tensor.get_plate(tensor.name)
        if self.rebar:
            hard_sample, relaxed_sample, cond_sample = split(
                tensor, plate, amt_slices=3
            )
            hard_cost, relaxed_cost, cond_cost = split(cost_node, plate, amt_slices=3)

        else:
            hard_sample = self._discretize(tensor, tensor.distribution)
            relaxed_sample = tensor
            cond_sample = self._conditional_gumbel_rsample(
                hard_sample, tensor.distribution
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
        d_param = storch.reduce_plates(d_param, plate_names=[tensor.name])
        # Compute backwards from the parameters using its total derivative
        if isinstance(param, storch.Tensor):
            param._tensor.backward(d_param._tensor, retain_graph=True)
        else:
            param.backward(d_param._tensor, retain_graph=True)
        # Compute the gradient variance
        variance = (d_param ** 2).sum(d_param.event_dim_indices())
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
