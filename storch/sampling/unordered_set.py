from typing import Optional

from torch.distributions import Distribution

from storch.sampling.swor import log1mexp, SampleWithoutReplacement
import storch

import torch


class UnorderedSet(SampleWithoutReplacement):
    def __init__(
        self,
        plate_name: str,
        k: int,
        comp_leave_two_out: bool = False,
        exact_integration: bool = False,
        num_int_points: int = 1000,
        a: float = 5.0,
        eos=None,
    ):
        super().__init__(plate_name, k, eos=eos)
        self.comp_leave_two_out = comp_leave_two_out
        self.exact_integration = exact_integration
        self.num_int_points = num_int_points
        self.a = a

    def weighting_function(
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

        if self.comp_leave_two_out:
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


class GumbelSoftmaxWOR(UnorderedSet):
    def __init__(
        self,
        plate_name: str,
        k: int,
        initial_temperature=1.0,
        min_temperature=1.0e-4,
        annealing_rate=1.0e-5,
        eos=None,
    ):
        super().__init__(plate_name, k, comp_leave_two_out=False, eos=eos)
        self.temperature = initial_temperature

    def sample(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        orig_distr_plates: [storch.Plate],
        requires_grad: bool,
    ) -> (torch.Tensor, storch.Plate):
        hard_sample, plate = super().sample(
            distr, parents, orig_distr_plates, requires_grad
        )
        from storch import conditional_gumbel_rsample

        gumbel_wor = conditional_gumbel_rsample(hard_sample, distr.probs,
                                                isinstance(distr, torch.distributions.Bernoulli), self.temperature)
        gumbel_wor = storch.StochasticTensor(
            gumbel_wor._tensor,
            hard_sample.parents,
            hard_sample.plates,
            hard_sample.name,
            self.k,
            distr,
            requires_grad,
        )
        return gumbel_wor, plate
