from __future__ import annotations
from typing import Optional, Callable, Tuple

import storch
import torch
from torch.distributions import Distribution, Gumbel
from storch.sampling.method import SamplingMethod
from storch.sampling.seq import IterDecoding, AncestralPlate, right_expand_as


class SampleWithoutReplacement(IterDecoding):
    """
    Sampling method for sampling without replacement from (sequences of) discrete distributions.
    Implements Stochastic Beam Search https://arxiv.org/abs/1903.06059 with the weighting as defined by
    REINFORCE without replacement https://openreview.net/forum?id=r1lgTGL5DE
    """

    EPS = 1e-8
    perturbed_log_probs: Optional[storch.Tensor] = None

    def __init__(self, plate_name: str, k: int, biased_iw: bool = False, eos=None):
        super().__init__(plate_name, k, eos)
        if k < 2:
            raise ValueError(
                "Can only sample with replacement for more than 1 samples."
            )
        self.biased_iw = biased_iw

    def reset(self):
        super().reset()
        # Cumulative perturbed log probabilities of the samples
        self.perturbed_log_probs = None

    def decode_step(
        self,
        indices: Tuple[int],
        yv_log_probs: storch.Tensor,
        joint_log_probs: Optional[storch.Tensor],
        sampled_support_indices: Optional[storch.Tensor],
        parent_indexing: Optional[storch.Tensor],
        is_conditional_sample: bool,
        amt_plates: int,
        amt_samples: int,
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        """
        Decode given the input arguments for a specific event using stochastic beam search.
        :param indices: Tuple of integers indexing the current event to sample.
        :param yv_log_probs:  Log probabilities of the different options for this event. distr_plates x k? x |D_yv|
        :param joint_log_probs: The log probabilities of the samples so far. None if `not is_conditional_sample`. prev_plates x amt_samples
        :param sampled_support_indices: Tensor of samples so far. None if this is the first set of indices. plates x k x events
        :param parent_indexing: Tensor indexing the parent sample. None if `not is_conditional_sample`.
        :param is_conditional_sample: True if a parent has already been sampled. This means the plates are more complex!
        :param amt_plates: The total amount of plates in both the distribution and the previously sampled variables
        :param amt_samples: The amount of active samples.
        :return: 3-tuple of `storch.Tensor`. 1: sampled_support_indices, with `:, indices` referring to the indices for the support.
        2: The updated `joint_log_probs` of the samples.
        3: The updated `parent_indexing`. How the samples index the parent samples. Can just return parent_indexing if nothing happens.
        4: The amount of active samples after this step.
        """

        first_sample = False
        if joint_log_probs is None:
            # We also know that k? is not present, so distr_plates x |D_yv|
            all_joint_log_probs = yv_log_probs
            # First condition on max being 0:
            self.perturbed_log_probs = 0.0
            first_sample = True
        elif is_conditional_sample > 0:
            # Make sure we are selecting the correct log-probabilities. As parents have been selected, this might change!
            # plates x amt_samples x |D_yv|
            yv_log_probs = yv_log_probs.gather(
                dim=-2,
                index=right_expand_as(
                    # Use the parent_indexing to select the correct plate samples. Make sure to limit to amt_samples!
                    parent_indexing[..., :amt_samples],
                    yv_log_probs,
                ),
            )
            # self.joint_log_probs: prev_plates x amt_samples
            # plates x amt_samples x |D_yv|
            all_joint_log_probs = joint_log_probs.unsqueeze(-1) + yv_log_probs
        else:
            # self.joint_log_probs: plates x amt_samples
            # plates x amt_samples x |D_yv|
            all_joint_log_probs = joint_log_probs.unsqueeze(
                -1
            ) + yv_log_probs.unsqueeze(-2)

        # Sample plates x k? x |D_yv| conditional Gumbel variables
        cond_G_yv = cond_gumbel_sample(all_joint_log_probs, self.perturbed_log_probs)

        # If there are finished samples, ensure eos is always sampled.
        if self.finished_samples is not None:
            # TODO: Is this the correct way of ensuring self.eos is always sampled for finished sequences?
            #  Coudl it bias things in any way?
            # Set the probability of continuing on finished sequences to -infinity so that they are filtered out during topk.
            # amt_finished
            finished_perturb_log_probs = self.perturbed_log_probs._tensor[
                self.finished_samples._tensor
            ]
            # amt_finished x |D_yv|
            finished_vec = finished_perturb_log_probs.new_full(
                (finished_perturb_log_probs.shape[0], cond_G_yv.shape[-1],),
                -float("inf"),
            )
            # Then make sure the log probability of the eos token is equal to the last perturbed log prob.
            finished_vec[:, self.eos] = finished_perturb_log_probs
            cond_G_yv[self.finished_samples] = finished_vec

        if not first_sample:
            # plates x (k * |D_yv|) (k == prev_amt_samples, in this case)
            cond_G_yv = cond_G_yv.reshape(cond_G_yv.shape[:-2] + (-1,))
            # Reshape log probs to plates x (k * |D_yv|). Matches perturbed shape.
            all_joint_log_probs = all_joint_log_probs.reshape(cond_G_yv.shape)

        # Select the samples given the perturbed log probabilities
        self.perturbed_log_probs, joint_log_probs, arg_top = self.select_samples(
            cond_G_yv, all_joint_log_probs
        )
        # Gather corresponding joint log probabilities.

        amt_samples = arg_top.shape[-1]

        if first_sample:
            # plates x amt_samples
            # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
            # (:) * amt_plates + (indices for events) + amt_samples
            indexing = (slice(None),) * amt_plates + (slice(0, amt_samples),) + indices
            sampled_support_indices[indexing] = arg_top
        else:
            joint_log_probs = all_joint_log_probs.reshape(cond_G_yv.shape).gather(
                dim=-1, index=arg_top
            )

            # |D_yv|
            size_domain = yv_log_probs.shape[-1]

            # Keep track of what parents were sampled for the arg top
            # plates x amt_samples
            chosen_parents = arg_top // size_domain
            sampled_support_indices = sampled_support_indices.gather(
                dim=amt_plates,
                index=right_expand_as(chosen_parents, sampled_support_indices),
            )
            if parent_indexing is not None:
                parent_indexing = parent_indexing.gather(dim=-1, index=chosen_parents)
            # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
            # plates x amt_samples
            chosen_samples = arg_top.remainder(size_domain)
            indexing = (slice(None),) * amt_plates + (slice(0, amt_samples),) + indices
            sampled_support_indices[indexing] = chosen_samples
        return sampled_support_indices, joint_log_probs, parent_indexing, amt_samples

    def select_samples(
        self, perturbed_log_probs: storch.Tensor, joint_log_probs: storch.Tensor,
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        """
        Given the perturbed log probabilities and the joint log probabilities of the new options, select which one to
        use for the sample.
        :param perturbed_log_probs: plates x (k? * |D_yv|). Perturbed log-probabilities. k is present if first_sample.
        :param joint_log_probs: plates x (k? * |D_yv|). Joint log probabilities of the options. k is present if first_sample.
        :param first_sample:
        :return: perturbed log probs of chosen samples, joint log probs of chosen samples, index of chosen samples
        """

        # We can sample at most the amount of what we previous sampled, combined with every option in the current domain
        # That is: prev_amt_samples * |D_yv|.
        amt_samples = min(self.k, perturbed_log_probs.shape[-1])
        # Take the top k over conditional perturbed log probs
        # plates x amt_samples
        perturbed_log_probs, arg_top = torch.topk(perturbed_log_probs, amt_samples, dim=-1)
        joint_log_probs = joint_log_probs.gather(dim=-1, index=arg_top)
        return perturbed_log_probs, joint_log_probs, arg_top


    def create_plate(self, plate_size: int, plates: [storch.Plate]) -> AncestralPlate:
        plate = super().create_plate(plate_size, plates)
        plate.perturb_log_probs = storch.Tensor(
            self.perturbed_log_probs._tensor,
            [self.perturbed_log_probs],
            self.perturbed_log_probs.plates + [plate],
        )
        return plate

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        # TODO: Doesnt take into account eos tokens
        # TODO: Does this add the plate to the weighting function result? Could be a big bug!
        return self.compute_iw(plate, self.biased_iw).detach()

    def compute_iw(self, plate: AncestralPlate, biased: bool):
        k = plate.perturb_log_probs.shape[-1]
        # Compute importance weights. The kth sample has 0 weight, and is only used to compute the importance weights
        # Equation 5
        q = (1 - torch.exp(
            - torch.exp(
                plate.log_probs - plate.perturb_log_probs._tensor[..., k - 1].unsqueeze(-1)
                ))).detach()
        iw = plate.log_probs.exp() / (q + self.EPS)
        # Set the weight of the kth sample (kappa) to 0.
        iw[..., k - 1] = 0.0
        if biased:
            # Equation 6 (normalization of importance weights)
            WS = storch.sum(iw, plate).detach()
            return iw / WS
        return iw

    def on_plate_already_present(self, plate: storch.Plate):
        if (
            not isinstance(plate, AncestralPlate)
            or plate.variable_index > self.variable_index
            or plate.n > self.k
        ):
            super().on_plate_already_present(plate)

    def set_mc_sample(
        self,
        new_sample_func: Callable[
            [Distribution, [storch.Tensor], [storch.Plate], int], torch.Tensor
        ],
    ) -> SamplingMethod:
        raise RuntimeError(
            "Cannot set monte carlo sampling for sampling without replacement."
        )


def log1mexp(a: torch.Tensor) -> torch.Tensor:
    """See appendix A of http://jmlr.org/papers/v21/19-985.html.
    Numerically stable implementation of log(1-exp(a))"""
    c = -0.693
    a1 = -a.abs()
    eps = 1e-6
    # exp_a = -a1.exp()
    # assert (exp_a >= -1).all()
    return torch.where(a1 > c, torch.log(-a1.expm1() + eps), torch.log1p(-a1.exp() + eps))


@storch.deterministic
def cond_gumbel_sample(all_joint_log_probs, perturbed_log_probs) -> torch.Tensor:
    # Sample plates x k? x |D_yv| Gumbel variables
    gumbel_d = Gumbel(loc=all_joint_log_probs, scale=1.0)
    G_yv = gumbel_d.rsample()

    # Condition the Gumbel samples on the maximum of previous samples
    # plates x k
    Z = G_yv.max(dim=-1)[0]
    T = perturbed_log_probs
    vi = T - G_yv + log1mexp(G_yv - Z.unsqueeze(-1))
    # plates (x k) x |D_yv|
    return T - vi.relu() - torch.nn.Softplus()(-vi.abs())


class SumAndSample(SampleWithoutReplacement):
    """
    Sums over S probable samples according to beam search and K sampled values that are not in the probable samples,
    then normalizes them accordingly.
    """

    def __init__(
        self,
        plate_name: str,
        sum_size: int,
        sample_size: int = 1,
        without_replacement: bool = False,
        eos=None,
    ):
        super().__init__(plate_name, sum_size + sample_size, eos=eos)
        self.sum_size = sum_size
        self.sample_size = sample_size
        if sum_size < 1 or sample_size < 1:
            raise ValueError("sum_size and sample_size should both be at least 1.")

    def select_samples(
        self, perturbed_log_probs: storch.Tensor, joint_log_probs: storch.Tensor,
    ) -> (storch.Tensor, storch.Tensor):
        # Select sum_size samples using joint log probs, and sample_size samples using perturbed joint log probs.

        # We can sample at most the amount of what we previous sampled, combined with every option in the current domain
        # That is: prev_amt_samples * |D_yv|.
        amt_sum = min(self.sum_size, joint_log_probs.shape[-1])
        # Take the top sum_size over the joint log probs. This is like beam search
        # plates x amt_samples
        _, sum_samples = torch.topk(joint_log_probs, amt_sum, dim=-1)
        sum_perturbed_log_probs = perturbed_log_probs[sum_samples]
        if amt_sum < self.sum_size:
            return sum_perturbed_log_probs, sum_samples

        # Not sure if this is the most efficient implementation
        # Should be positive, by the previous conditional.
        amt_sample = min(
            self.sample_size, perturbed_log_probs.shape[-1] - self.sum_size
        )
        sample_perturbed_log_probs, samples = torch.topk(
            joint_log_probs, amt_sample + perturbed_log_probs.shape[-1], dim=-1
        )

        # TODO: This isn't finished yet.
