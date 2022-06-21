import torch
from storch.sampling.seq import AncestralPlate

import storch
from typing import Optional, Tuple

from storch import StochasticTensor, CostTensor

from storch.method.multi_sample_reinforce import SeqMethod
from storch.sampling import RaoBlackwellizedSample


class RaoBlackwellSF(SeqMethod):
    def __init__(self, plate_name, n_samples):
        super().__init__(plate_name, RaoBlackwellizedSample(plate_name, n_samples))

    def estimator(
        self, tensor: StochasticTensor, cost: CostTensor
    ) -> Tuple[Optional[storch.Tensor], Optional[storch.Tensor]]:
        # Only uses the last sample for the score function
        zeros = torch.zeros_like(cost, dtype=tensor.dtype)
        cost_plate: AncestralPlate = tensor.get_plate(self.plate_name)

        if cost_plate.n == self.sampling_method.k:
            p_index = tensor.get_plate_dim_index(self.plate_name)
            slize = [slice(None)] * p_index
            slize.append(self.sampling_method.k - 1)
            slize = tuple(slize)
            # Find the correct index for the last sample
            zeros[slize] = cost_plate.log_probs._tensor[slize]

        return zeros, None