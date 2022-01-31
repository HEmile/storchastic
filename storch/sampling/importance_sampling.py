from typing import Optional, Dict, Callable, List

import torch
from storch.sampling.seq import MCDecoder, AncestralPlate

from storch.util import get_distr_parameters
from torch.distributions import Distribution

import storch
from storch import Plate
from storch.sampling import SamplingMethod, MonteCarlo

def add_prop_plates(proposal: Distribution, plates: List[Plate]):
    # Ensure proposal plates are applied to sample
    params: Dict[str, storch.Tensor] = get_distr_parameters(
        proposal, filter_requires_grad=False
    )
    for name, p in params.items():
        if isinstance(p, storch.Tensor):
            # The sample should have the batch links of the parameters in the distribution, if present.
            for plate in p.plates:
                if plate not in plates:
                    # Insert plate after newly created plate
                    plates.insert(1, plate)


class ImportanceSampling(SamplingMethod):
    # Performs importance sampling from a single distribution
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
        return (true_prob / (plate.n * proposal_prob)).detach()

    def sample(self, distr: Distribution, parents: [storch.Tensor], plates: [Plate], requires_grad: bool) -> (
        storch.StochasticTensor, Plate):
        storch_tensor, new_plate = self._method.sample(distr, parents, plates, requires_grad)

        add_prop_plates(self.proposal, storch_tensor.plates)

        return storch_tensor, new_plate

        # return self._method.sample(distr, parents, plates, requires_grad)

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        return self._method.weighting_function(tensor, plate)


class ImportanceSampleDecoder(MCDecoder):
    # For sampling sequences of random variables, where each RV is sampled from an importance sampling distribution
    def __init__(self, plate_name: str, k: int, is_callback: Callable[[List[storch.StochasticTensor]], Distribution],
                 eos=None, epsilon=1e-7):
        """
        :param is_callback: Called every time sample is called. Pass the new proposal distribution here. It gives you
          the list of samples for previous RVs in the sequence.
        """
        super().__init__(plate_name, k, eos)
        self.is_callback = is_callback
        self.proposal_dists: List[Distribution] = []
        self.epsilon = torch.tensor(epsilon)

    def reset(self):
        super().reset()
        self.proposal_dists = []

    def decode(
        self,
        distribution: Distribution,
        joint_log_probs: Optional[storch.Tensor],
        parents: [storch.Tensor],
        orig_distr_plates: [storch.Plate],
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        proposal = self.is_callback(self.seq)
        add_prop_plates(proposal, orig_distr_plates)
        self.proposal_dists.append(proposal)
        return super().decode(distribution, joint_log_probs, parents, orig_distr_plates)

    def mc_sample(
            self,
            distr: Distribution,
            parents: [storch.Tensor],
            plates: [Plate],
            amt_samples: int,
    ) -> torch.Tensor:
        proposal = self.proposal_dists[-1]
        return super().mc_sample(proposal, parents, plates, amt_samples)

    def weighting_function(
            self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        # Compute importance weights
        assert isinstance(plate, AncestralPlate)
        true_prob = tensor.distribution.log_prob(tensor).exp()
        proposal_prob = self.proposal_dists[plate.variable_index].log_prob(tensor).exp()
        parent_weight = 1.0
        if plate.parent_plate:
            parent_weight = plate.parent_plate.weight * plate.n
        # TODO: Should I detach the weights?
        return (parent_weight * true_prob / (plate.n * (proposal_prob + self.epsilon))).detach()


