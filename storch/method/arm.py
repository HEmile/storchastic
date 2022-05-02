from typing import Optional, Tuple

from torch.distributions import (
    Distribution,
    Bernoulli,
    TransformedDistribution,
    Uniform,
    SigmoidTransform,
    AffineTransform,
)
import torch
import storch
from storch import Plate, StochasticTensor, CostTensor
from storch.method import Method
from storch.sampling import SamplingMethod, MonteCarlo
from storch.util import magic_box


class Logistic(TransformedDistribution):
    def __init__(self, loc: float, scale: float):
        # Copied from https://pytorch.org/docs/stable/distributions.html#torch.distributions.transformed_distribution.TransformedDistribution
        super().__init__(
            Uniform(0, 1),
            transforms=[SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)],
        )


class ARM(Method):
    def __init__(
        self,
        plate_name: str,
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
    ):
        if not sampling_method:
            sampling_method = MonteCarlo(plate_name, n_samples)
        super().__init__(
            plate_name, sampling_method.set_mc_sample(self.sample_arm).set_mc_weighting_function(self.weighting_function),
        )

    def sample_arm(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        amt_samples: int,
    ):
        if not isinstance(distr, Bernoulli):
            raise ValueError("ARM only works for the Bernoulli distribution.")
        logits = distr.logits
        eps = Logistic(0.0, 1.0).sample((amt_samples,) + logits.shape)
        z = eps + logits
        z_tilde = -eps + logits
        return torch.cat([z, z_tilde], dim=0)

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        # The antithetic sample is not according to the true distribution, so we cannot count it during the weighting
        n = int(tensor.n / 2)
        weighting = tensor._tensor.new_zeros((tensor.n,))
        weighting[:n] = tensor._tensor.new_tensor(1.0 / n)
        return weighting

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        return (tensor > 0.0).float()

    def comp_estimator(
        self, tensor: torch.Tensor, cost: torch.Tensor, logits: torch.Tensor, plate_index: int, n: int
    ) -> Tuple[storch.Tensor, storch.Tensor]:
        _index_z = (slice(None),) * plate_index + (slice(n),)
        _index_z_tilde = (slice(None),) * plate_index + (slice(n, 2 * n),)
        z = tensor[_index_z]

        baseline = torch.zeros_like(cost)
        baseline[_index_z] = 0.5 * (cost[_index_z] + cost[_index_z_tilde])

        logistic = Logistic(logits, 1.0)
        log_prob = torch.zeros_like(tensor)
        log_prob[_index_z] = logistic.log_prob(z.detach())
        return log_prob, baseline

    def estimator(
        self, tensor: StochasticTensor, cost: CostTensor
    ) -> Tuple[
        Optional[storch.Tensor], Optional[storch.Tensor]
    ]:
        # TODO: No support for alternative plate weighting
        plate = tensor.get_plate(tensor.name)
        index = tensor.get_plate_dim_index(plate.name)
        log_prob, baseline = storch.deterministic(self.comp_estimator)(
            tensor, cost, tensor.distribution.logits, index, plate.n // 2
        )

        return log_prob, (1-magic_box(log_prob)) * baseline


class DisARM(Method):
    """
    Introduces by Dong, Mnih and Tucker, 2020 https://arxiv.org/abs/2006.10680
    """

    def __init__(
        self,
        plate_name: str,
        sampling_method: Optional[SamplingMethod] = None,
        n_samples: int = 1,
    ):
        if not sampling_method:
            sampling_method = MonteCarlo(plate_name, n_samples)
        super().__init__(
            plate_name, sampling_method.set_mc_sample(self.sample_disarm),
        )

    def sample_disarm(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        amt_samples: int,
    ):
        if not isinstance(distr, Bernoulli):
            raise ValueError("DisARM only works for the Bernoulli distribution.")
        logits = distr.logits
        eps = Logistic(0.0, 1.0).sample((amt_samples,) + logits.shape)
        z = eps + logits
        z_tilde = -eps + logits
        # Sample (b, \tilde{b})
        return torch.cat([z, z_tilde], dim=0) > 0.0

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        # The antithetic sample is not according to the true distribution, so we cannot count it during the weighting
        n = int(tensor.n / 2)
        weighting = tensor._tensor.new_zeros((tensor.n,))
        weighting[:n] = tensor._tensor.new_tensor(1.0 / n)
        return weighting

    @storch.deterministic
    def comp_estimator(
        self, tensor: StochasticTensor, cost: CostTensor, logits, plate_index, n
    ) -> Tuple[storch.Tensor, storch.Tensor]:
        _index_z = (None,) * plate_index + (slice(n),)
        _index_ztilde = (None,) * plate_index + (slice(n, 2 * n),)
        f_z_tilde = cost[_index_ztilde]
        b = tensor[_index_z]
        b_tilde = tensor[_index_ztilde]

        weighting = ((-1) ** b_tilde.float()) * ~b.eq(b_tilde) * logits.abs().sigmoid()
        log_prob = torch.zeros_like(tensor, dtype=cost.dtype)
        log_prob[_index_z] = weighting.detach() * logits

        baseline = torch.zeros_like(cost)
        baseline[_index_z] = f_z_tilde

        return log_prob, baseline

    def estimator(
        self, tensor: StochasticTensor, cost: CostTensor
    ) -> Tuple[
        Optional[storch.Tensor], Optional[storch.Tensor]
    ]:
        # TODO: No support for alternative plate weighting
        # TODO: This doesn't follow the current implementation
        plate = tensor.get_plate(tensor.name)
        index = tensor.get_plate_dim_index(plate.name)
        multiplicative, baseline = self.comp_estimator(
            tensor, cost, tensor.distribution.logits, index, plate.n // 2
        )

        return (
            0.5 * multiplicative.sum(dim=multiplicative.event_dim_indices),
            baseline,
        )
