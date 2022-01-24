from typing import Optional, Dict

import torch
from storch.util import get_distr_parameters
from torch.distributions import Distribution

import storch
from storch import Plate
from storch.sampling import SamplingMethod, MonteCarlo


class ImportanceSampling(SamplingMethod):
    def __init__(self, plate_name: str, n: int, proposal: Distribution):
        super().__init__(plate_name)
        self._method = MonteCarlo(plate_name, n)
        self.proposal = proposal
        self._method.set_mc_sample(self.mc_sample)
        self._method.set_mc_weighting_function(self.mc_weighting_function)

    def mc_sample(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        amt_samples: int,
    ) -> torch.Tensor:
        # Sample from proposal instead of actual distribution
        return self.proposal.sample((amt_samples,))

    def mc_weighting_function(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        # Apply importance weights
        true_prob = tensor.distribution.log_prob(tensor).exp()
        proposal_prob = self.proposal.log_prob(tensor).exp()
        return true_prob / proposal_prob

    def sample(self, distr: Distribution, parents: [storch.Tensor], plates: [Plate], requires_grad: bool) -> (
        storch.StochasticTensor, Plate):
        storch_tensor, new_plate = self._method.sample(distr, parents, plates, requires_grad)
        # Ensure proposal plates are applied to sample
        params: Dict[str, storch.Tensor] = get_distr_parameters(
            self.proposal, filter_requires_grad=False
        )
        for name, p in params.items():
            if isinstance(p, storch.Tensor):
                # The sample should have the batch links of the parameters in the distribution, if present.
                for plate in p.plates:
                    if plate not in storch_tensor.plates:
                        # Insert plate after newly created plate
                        storch_tensor.plates.insert(1, plate)
        return storch_tensor, new_plate

        # return self._method.sample(distr, parents, plates, requires_grad)

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        return self._method.weighting_function(tensor, plate)