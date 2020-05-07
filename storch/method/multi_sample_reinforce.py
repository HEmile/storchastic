from typing import Optional

import torch
import storch

from storch.method.sampling import SampleWithoutReplacementMethod, AncestralPlate


class ScoreFunctionWOR(SampleWithoutReplacementMethod):
    """
    Implement Buy 4 REINFORCE Samples, Get a Baseline for Free! https://openreview.net/pdf?id=r1lgTGL5DE
    Use biased=True for the biased normalized version which has lower variance.
    """

    EPS = 1e-8

    def __init__(
        self, plate_name: str, k: int, biased: bool = True, use_baseline: bool = True
    ):
        # Use k + 1 to be able to compute kappa, the k+1th perturbed log-prob
        super().__init__(plate_name, k + 1)
        self.biased = biased
        self.use_baseline = use_baseline

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        return self._compute_iw(plate, self.biased).detach()

    def _compute_iw(self, plate: AncestralPlate, biased: bool):
        # Compute importance weights. The kth sample has 0 weight, and is only used to compute the importance weights
        q = (
            1
            - (
                -(
                    plate.log_probs
                    - plate.perturb_log_probs._tensor[..., self.k - 1].unsqueeze(-1)
                ).exp()
            ).exp()
        ).detach()
        iw = plate.log_probs.exp() / (q + self.EPS)
        # It's probably not going to like this? Would iw._tensor work?
        # Set the weight of the kth sample (kappa) to 0.
        iw[..., self.k - 1] = 0.0
        if biased:
            WS = storch.sum(iw, plate).detach()
            return iw / WS
        return iw

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

    def estimator(
        self, tensor: storch.StochasticTensor, cost_node: storch.CostTensor
    ) -> Optional[storch.Tensor]:
        cost_plate = None
        for _p in cost_node.plates:
            if _p.name == self.plate_name:
                cost_plate = _p
                break
        if self.use_baseline:
            iw = self._compute_iw(cost_plate, biased=False)
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
            iw = self._compute_iw(cost_plate, self.biased)
            return storch.sum(cost_node.detach() * iw, self.plate_name)
