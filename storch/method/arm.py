from typing import Optional

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
from storch import Plate
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

    def estimator(
        self, tensor: storch.StochasticTensor, cost_node: storch.CostTensor
    ) -> Optional[storch.Tensor]:
        # TODO: No support for alternative plate weighting
        plate = tensor.get_plate(tensor.name)
        f_z, f_z_tilde = storch.util.split(cost_node, plate, amt_slices=2)
        z, _ = storch.util.split(tensor, plate, amt_slices=2)

        avg_cost = 0.5 * (f_z - f_z_tilde)
        logistic = Logistic(tensor.distribution.logits, 1.0)
        log_prob = logistic.log_prob(z.detach())
        log_prob = log_prob.sum(dim=log_prob.event_dim_indices)
        return avg_cost.detach() * log_prob
