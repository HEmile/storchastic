from __future__ import annotations
from typing import Optional, Union, List, Callable

import storch
import torch
from torch.distributions import Distribution, Gumbel
import itertools
from storch.sampling.seq import SequenceDecoding, AncestralPlate


class SampleWithoutReplacement(SequenceDecoding):
    """
    Sampling method for sampling without replacement from (sequences of) discrete distributions.
    Implements Stochastic Beam Search https://arxiv.org/abs/1903.06059 with the weighting as defined by
    REINFORCE without replacement https://openreview.net/forum?id=r1lgTGL5DE
    """

    EPS = 1e-8

    def __init__(self, plate_name: str, k: int, biased_iw: bool = False):
        super().__init__(plate_name, k)
        if k < 2:
            raise ValueError(
                "Can only sample with replacement for more than 1 samples."
            )
        self.biased_iw = biased_iw

    def reset(self):
        super().reset()
        # Cumulative perturbed log probabilities of the samples
        self.perturbed_log_probs = None

    def decode(
        self,
        d_log_probs: storch.Tensor,
        support: storch.Tensor,
        joint_log_probs: Optional[storch.Tensor],
        is_conditional_sample: bool,
        amt_plates: int,
        event_shape: torch.Size,
        element_shape: torch.Size,
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        """
        Decode using stochastic beam search.
        :param d_log_probs: Log probability given by the distribution. distr_plates x k? x |D_yv| x events
        :param support: The support of this distribution. plates x |D_yv| x events x event_shape
        :param joint_log_probs: The log probabilities of the samples so far. prev_plates x amt_samples
        :param is_conditional_sample: True if a parent has already been sampled. This means the plates are more complex!
        :param amt_plates: The total amount of plates in both the distribution and the previously sampled variables
        :param event_shape: The shape of the conditionally independent events
        :param element_shape: The shape of a domain element. (|D_yv|,) for `torch.distributions.OneHotCategorical`, otherwise (,).
        :return: 3-tuple of `storch.Tensor`. 1: The sampled value. 2: The new joint log probabilities of the samples.
        3: How the samples index the parent samples. Can just be a range if there is no choosing happening.
        """
        ranges = []
        for size in event_shape:
            ranges.append(list(range(size)))

        amt_samples = 0
        parent_indexing = None
        if joint_log_probs is not None:
            # Initialize a tensor (self.parent_indexing) that keeps track of what samples link to previous choices of samples
            # Note that joint_log_probs.shape[-1] is amt_samples, not k. It's possible that amt_samples < k!
            amt_samples = joint_log_probs.shape[-1]
            # plates x k
            parent_indexing = support.new_zeros(
                size=support.shape[:amt_plates] + (self.k,), dtype=torch.long
            )

            # probably can go wrong if plates are missing.
            parent_indexing[..., :amt_samples] = left_expand_as(
                torch.arange(amt_samples), parent_indexing
            )
        # plates x k x events
        sampled_support_indices = support.new_zeros(
            size=support.shape[:amt_plates]  # plates
            + (self.k,)
            + support.shape[
                amt_plates + 1 : -len(element_shape) if len(element_shape) > 0 else None
            ],  # events
            dtype=torch.long,
        )
        # Sample independent tensors in sequence
        # Iterate over the different (conditionally) independent samples being taken (the events)
        for indices in itertools.product(*ranges):
            # Log probabilities of the different options for this sample step (event)
            # distr_plates x k? x |D_yv|
            yv_log_probs = d_log_probs[(...,) + indices]
            first_sample = False
            if joint_log_probs is None:
                # We also know that k? is not present, so distr_plates x |D_yv|
                all_joint_log_probs = yv_log_probs
                # First condition on max being 0:
                self.perturbed_log_probs = 0.0
                first_sample = True
            elif is_conditional_sample:
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
            cond_G_yv = cond_gumbel_sample(
                all_joint_log_probs, self.perturbed_log_probs
            )
            if first_sample:
                # No parent has been sampled yet
                # shape(cond_G_yv) is plates x |D_yv|
                amt_samples = min(self.k, cond_G_yv.shape[-1])
                # Compute top k over the conditional perturbed log probs
                # plates x amt_samples
                self.perturbed_log_probs, arg_top = torch.topk(
                    cond_G_yv, amt_samples, dim=-1
                )
                # plates x amt_samples
                joint_log_probs = all_joint_log_probs.gather(dim=-1, index=arg_top)
                # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
                # (:) * amt_plates + (indices for events) + amt_samples
                indexing = (
                    (slice(None),) * amt_plates + (slice(0, amt_samples),) + indices
                )
                sampled_support_indices[indexing] = arg_top
            else:
                # plates x (k * |D_yv|) (k == prev_amt_samples, in this case)
                cond_G_yv = cond_G_yv.reshape(cond_G_yv.shape[:-2] + (-1,))
                # We can sample at most the amount of what we previous sampled, combined with every option in the current domain
                # That is: prev_amt_samples * |D_yv|. But we also want to limit by k.
                amt_samples = min(self.k, cond_G_yv.shape[-1])
                # Take the top k over conditional perturbed log probs
                # plates x amt_samples
                self.perturbed_log_probs, arg_top = torch.topk(
                    cond_G_yv, amt_samples, dim=-1
                )
                # Gather corresponding joint log probabilities. First reshape like previous to plates x (k * |D_yv|).
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
                    parent_indexing = parent_indexing.gather(
                        dim=-1, index=chosen_parents
                    )
                # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
                # plates x amt_samples
                chosen_samples = arg_top.remainder(size_domain)
                indexing = (
                    (slice(None),) * amt_plates + (slice(0, amt_samples),) + indices
                )
                sampled_support_indices[indexing] = chosen_samples

        # Finally, index the support using the sampled indices to get the sample!
        if amt_samples < self.k:
            # plates x amt_samples x events
            sampled_support_indices = sampled_support_indices[
                (...,) + (slice(amt_samples),) + (slice(None),) * len(ranges)
            ]
        expanded_indices = right_expand_as(sampled_support_indices, support)
        sample = support.gather(dim=amt_plates, index=expanded_indices)
        return sample, joint_log_probs, parent_indexing

    def create_plate(self, plate_size: int, plates: [storch.Plate]) -> AncestralPlate:
        plate = super().create_plate(plate_size, plates)
        plate.perturb_log_probs = storch.Tensor(
            self.perturb_log_probs._tensor,
            [self.perturb_log_probs],
            self.perturb_log_probs.plates + [plate],
        )
        return plate

    def plate_weighting(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        return self.compute_iw(plate, self.biased_iw).detach()

    def compute_iw(self, plate: AncestralPlate, biased: bool):
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
        # Set the weight of the kth sample (kappa) to 0.
        iw[..., self.k - 1] = 0.0
        if biased:
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
    return torch.where(a1 > c, torch.log(-a1.expm1()), torch.log1p(-a1.exp()))


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


@storch.deterministic(l_broadcast=False)
def right_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(...,) + (None,) * diff].expand(
        (-1,) * tensor.ndim + expand_as.shape[tensor.ndim :]
    )


def left_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(None,) * diff].expand(expand_as.shape[:diff] + (-1,) * tensor.ndim)
