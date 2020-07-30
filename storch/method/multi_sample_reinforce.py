from typing import Optional

import storch
from storch.method.method import Method

from storch.sampling import SampleWithoutReplacement


class ScoreFunctionWOR(Method):
    """
    Implement Buy 4 REINFORCE Samples, Get a Baseline for Free! https://openreview.net/pdf?id=r1lgTGL5DE
    Use biased=True for the biased normalized version which has lower variance.
    """

    def __init__(
        self, plate_name: str, k: int, biased: bool = True, use_baseline: bool = True
    ):
        # Use k + 1 to be able to compute kappa, the k+1th perturbed log-prob
        super().__init__(plate_name, SampleWithoutReplacement(plate_name, k, biased))
        self.biased = biased
        self.use_baseline = use_baseline

    def adds_loss(
        self, tensor: storch.StochasticTensor, cost_node: storch.CostTensor
    ) -> bool:
        # We only want to add a loss on the stochastic tensor with the same plate as the cost node.
        # This is because the estimator computes the gradient with the respect to the JOINT log probability.
        # If we would have added the gradient for all stochastic tensors, these would just be duplicates of the same
        # loss being added (ie that gradient would be oversampled)
        for distr_plate in tensor.plates:
            if distr_plate.name == self.plate_name:
                for cost_plate in cost_node.plates:
                    if cost_plate.name == self.plate_name:
                        if cost_plate is distr_plate:
                            return True
                        return False
                raise ValueError(
                    "The given tensor contains an ancestral plate that the cost node doesn't have."
                )
        return False

    def multiplicative_estimator(
        self, tensor: storch.StochasticTensor, cost_node: storch.CostTensor
    ) -> Optional[storch.Tensor]:
        cost_plate = None
        for _p in cost_node.plates:
            if _p.name == self.plate_name:
                cost_plate = _p
                break
        if self.use_baseline:
            iw = self.sampling_method.compute_iw(cost_plate, biased=False)
            BS = storch.sum(iw * cost_node, cost_plate)
            probs = cost_plate.log_probs.exp()
            if self.biased:
                # Equation 11
                WS = storch.sum(iw, cost_plate)
                WiS = (WS - iw + probs).detach()
                diff_cost = cost_node - BS / WS
                return storch.sum(iw / WiS * diff_cost.detach(), cost_plate)
            else:
                # Equation 10
                weighted_cost = cost_node * (1 - probs + iw)
                diff_cost = weighted_cost - BS
                return storch.sum(iw * diff_cost.detach(), cost_plate)
        else:
            # Equation 9
            # TODO: This seems inefficient... The plate should already contain the IW, right? Same for above if not self.biased
            iw = self.sampling_method.compute_iw(cost_plate, self.biased)
            return storch.sum(cost_node.detach() * iw, self.plate_name)
