from typing import Optional

import torch
import storch
from storch.method.sampling import (
    SampleWithoutReplacementMethod,
    AncestralPlate,
    log1mexp,
)


class UnorderedSetEstimator(SampleWithoutReplacementMethod):
    def __init__(
        self,
        plate_name: str,
        k: int,
        use_baseline: bool = True,
        exact_integration=False,
        num_int_points=1000,
        a=5.0,
    ):
        super().__init__(plate_name, k)
        self.use_baseline = use_baseline
        self.exact_integration = exact_integration
        self.num_int_points = num_int_points
        self.a = a

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        # plates_w_k is a sequence of plates of which one is the input ancestral plate

        # Computes p(s) * R(S^k, s), or the probability of the sample times the leave-one-out ratio.
        # For details, see https://openreview.net/pdf?id=rklEj2EFvB
        # Code based on https://github.com/wouterkool/estimating-gradients-without-replacement/blob/master/bernoulli/gumbel.py
        log_probs = plate.log_probs.detach()

        # Compute integration points for the trapezoid rule: v should range from 0 to 1, where both v=0 and v=1 give a value of 0.
        # As the computation happens in log-space, take the logarithm of the result.
        # N
        v = (
            torch.arange(1, self.num_int_points, out=log_probs._tensor.new())
            / self.num_int_points
        )
        log_v = v.log()

        # Compute log(1-v^{exp(log_probs+a)}) in a numerically stable way in log-space
        # Uses the gumbel_log_survival function from
        # https://github.com/wouterkool/estimating-gradients-without-replacement/blob/master/bernoulli/gumbel.py
        # plates_w_k x N
        g_bound = (
            log_probs[..., None]
            + self.a
            + torch.log(-log_v)[log_probs.plate_dims * (None,) + (slice(None),)]
        )

        # Gumbel log survival: log P(g > g_bound) = log(1 - exp(-exp(-g_bound))) for standard gumbel g
        # If g_bound >= 10, use the series expansion for stability with error O((e^-10)^6) (=8.7E-27)
        # See https://www.wolframalpha.com/input/?i=log%281+-+exp%28-y%29%29
        y = torch.exp(g_bound)
        # plates_w_k x N
        terms = torch.where(
            g_bound >= 10, -g_bound - y / 2 + y ** 2 / 24 - y ** 4 / 2880, log1mexp(y)
        )

        # Compute integrands (without subtracting the special value s)
        # plates x N
        sum_of_terms = storch.sum(terms, plate)
        phi_S = storch.logsumexp(log_probs, plate)
        phi_D_min_S = log1mexp(phi_S)

        # plates x N
        integrand = (
            sum_of_terms
            + torch.expm1(self.a + phi_D_min_S)[..., None]
            * log_v[phi_D_min_S.plate_dims * (None,) + (slice(None),)]
        )

        # Subtract one term the for element that is left out in R
        # Automatically unsqueezes correctly using plate dimensions
        # plates_w_k x N
        integrand_without_s = integrand - terms

        # plates
        log_p_S = integrand.logsumexp(dim=-1)
        # plates_w_k
        log_p_S_without_s = integrand_without_s.logsumexp(dim=-1)

        # plates_w_k
        log_leave_one_out = log_p_S_without_s - log_p_S

        if self.use_baseline:
            # Compute the integrands for the 2nd order leave one out ratio.
            # Make sure to properly choose the indices: We shouldn't subtract the same term twice on the diagonals.
            # k x k
            skip_diag = storch.Tensor(
                1 - torch.eye(plate.n, out=log_probs._tensor.new()), [], [plate]
            )
            # plates_w_k x k x N
            integrand_without_ss = (
                integrand_without_s[..., None, :]
                - terms[..., None, :] * skip_diag[..., None]
            )
            # plates_w_k x k
            log_p_S_without_ss = integrand_without_ss.logsumexp(dim=-1)

            plate.log_snd_leave_one_out = log_p_S_without_ss - log_p_S_without_s

        # Return the unordered set estimator weighting
        return (log_leave_one_out + log_probs).exp().detach()

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
        # Note: We automatically multiply with leave-one-out ratio in the plate reduction
        plate = None
        for _p in cost_node.plates:
            if _p.name == self.plate_name:
                plate = _p
                break
        if not self.use_baseline:
            return plate.log_probs * cost_node.detach()

        # Subtract the 'average' cost of other samples, keeping in mind that samples are not independent.
        # plates x k
        baseline = storch.sum(
            (plate.log_probs + plate.log_snd_leave_one_out).exp() * cost_node, plate
        )
        # Make sure the k dimension is recognized as batch dimension
        baseline = storch.Tensor(
            baseline._tensor, [baseline], baseline.plates + [plate]
        )
        advantage = cost_node - baseline
        return plate.log_probs * advantage.detach()
