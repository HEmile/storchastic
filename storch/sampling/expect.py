from typing import Optional

from storch.sampling import SamplingMethod
from torch.distributions import Distribution
import storch
import torch
from storch import Plate
import itertools


class Enumerate(SamplingMethod):
    def __init__(self, plate_name: str, budget=10000):
        super().__init__(plate_name)
        self.budget = budget

    def sample(
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

        s_tensor = storch.StochasticTensor(
            enumerate_tensor,
            parents,
            plates,
            self.plate_name,
            plate_size,
            distr,
            requires_grad,
        )
        return s_tensor, plate

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        # Weight by the probability of each possible event
        log_probs = tensor.distribution.log_prob(tensor)
        if log_probs.plate_dims < len(log_probs.shape):
            log_probs = log_probs.sum(
                dim=list(range(tensor.plate_dims, len(log_probs.shape)))
            )
        return log_probs.exp()
