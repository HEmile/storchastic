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


class Logistic(TransformedDistribution):
    def __init__(self, loc, scale):
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
            plate_name, sampling_method.set_mc_sample(self.sample_arm),
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

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        # The antithetic sample is not according to the true distribution, so we cannot count it during the weighting
        n = int(tensor.n / 2)
        weighting = tensor._tensor.new_zeros((tensor.n,))
        weighting[:n] = tensor._tensor.new_tensor(1.0 / n)
        return weighting

    def post_sample(self, tensor: storch.StochasticTensor) -> Optional[storch.Tensor]:
        return tensor > 0.0

    def adds_loss(
        self, tensor: storch.StochasticTensor, cost_node: storch.CostTensor
    ) -> bool:
        return True

    @storch.deterministic
    def comp_estimator(
        self, tensor: StochasticTensor, cost: CostTensor, logits, plate_index, n
    ) -> Tuple[storch.Tensor, storch.Tensor]:
        _index1 = (None,) * plate_index + (slice(n, 2 * n),)
        _index2 = (None,) * plate_index + (slice(n),)
        f_z_tilde = cost[_index1]
        z = tensor[_index2]

        baseline = torch.zeros_like(cost)
        baseline[_index2] = f_z_tilde

        logistic = Logistic(logits, 1.0)
        log_prob = torch.zeros_like(tensor)
        log_prob[_index2] = logistic.log_prob(z.detach())
        return log_prob, baseline

    def estimator(
        self, tensor: StochasticTensor, cost: CostTensor
    ) -> Tuple[
        Optional[storch.Tensor], Optional[storch.Tensor], Optional[storch.Tensor]
    ]:
        # TODO: No support for alternative plate weighting
        plate = tensor.get_plate(tensor.name)
        index = tensor.get_plate_dim_index(plate.name)
        log_prob, baseline = self.comp_estimator(
            tensor, cost, tensor.distribution.logits, index, plate.n // 2
        )

        return 0.5 * log_prob.sum(log_prob.event_dim_indices), baseline, None


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

    def adds_loss(
        self, tensor: storch.StochasticTensor, cost_node: storch.CostTensor
    ) -> bool:
        return True

    @storch.deterministic
    def comp_estimator(
        self, tensor: StochasticTensor, cost: CostTensor, logits, plate_index, n
    ) -> Tuple[storch.Tensor, storch.Tensor]:
        _index1 = (None,) * plate_index + (slice(n),)
        _index2 = (None,) * plate_index + (slice(n, 2 * n),)
        f_z_tilde = cost[_index2]
        b = tensor[_index1]
        b_tilde = tensor[_index2]

        weighting = ((-1) ** b_tilde.float()) * ~b.eq(b_tilde) * logits.abs().sigmoid()
        log_prob = torch.zeros_like(tensor, dtype=cost.dtype)
        log_prob[_index1] = weighting.detach() * logits

        baseline = torch.zeros_like(cost)
        baseline[_index1] = f_z_tilde

        return log_prob, baseline

    def estimator(
        self, tensor: StochasticTensor, cost: CostTensor
    ) -> Tuple[
        Optional[storch.Tensor], Optional[storch.Tensor], Optional[storch.Tensor]
    ]:
        # TODO: No support for alternative plate weighting
        plate = tensor.get_plate(tensor.name)
        index = tensor.get_plate_dim_index(plate.name)
        multiplicative, baseline = self.comp_estimator(
            tensor, cost, tensor.distribution.logits, index, plate.n // 2
        )

        return (
            0.5 * multiplicative.sum(dim=multiplicative.event_dim_indices),
            baseline,
            None,
        )
