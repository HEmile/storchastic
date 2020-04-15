from __future__ import annotations
from typing import Optional

import storch
import torch
from torch.distributions import Distribution, Gumbel
import itertools
from storch.method.method import Method


class SampleWithoutReplacementMethod(Method):
    def __init__(self, plate_name: str, k: int):
        super().__init__(plate_name)
        if k < 2:
            raise ValueError(
                "Can only sample with replacement for more than 1 samples."
            )
        self.k = k
        self.reset()

    def reset(self):
        self.log_probs = None
        self.perturbed_log_probs = None
        self.selected_samples = None
        self.variable_index = 0
        self.last_plate = None

    def _sample_tensor(
        self, distr: Distribution, parents: [storch.Tensor], plates: [storch.Plate]
    ) -> (torch.Tensor, int):
        # Perform stochastic beam search given the log-probs and perturbed log probs so far.
        # Samples k values from this distribution so that all total configurations are unique.
        # TODO: We have to keep track what samples are taken at each step for the backward pass.
        # Why? We need to think about what samples are discarded at some point because they are pruned away.
        # In the estimator they will still appear! So we'll have to think about that. They don't deserve a gradient
        # as they are only partial configurations and thus we don't know their loss.
        (
            samples,
            self.log_probs,
            self.perturbed_log_probs,
            self.selected_samples,
        ) = stochastic_beam_search(
            distr, self.k, len(plates), self.log_probs, self.perturbed_log_probs,
        )
        self.variable_index += 1
        return samples

    def _create_plate(
        self,
        sampled_tensor: torch.Tensor,
        other_plates: [storch.Plate],
        plate_size: int,
    ) -> storch.Plate:
        for plate in other_plates:
            if plate.name != self.plate_name:
                continue

        self.last_plate = AncestralPlate(
            self.plate_name,
            plate_size,
            self.variable_index,
            self.last_plate,
            self.selected_samples,
            None,
        )
        return self.last_plate


class AncestralPlate(storch.Plate):
    def __init__(
        self,
        name: str,
        n: int,
        variable_index: int,
        parent_plate: AncestralPlate,
        selected_samples: storch.Tensor,
        weight: Optional[storch.Tensor] = None,
    ):
        super().__init__(name, n, weight)
        assert (not parent_plate and variable_index == 1) or parent_plate.n <= self.n
        self.parent_plate = parent_plate
        self.selected_samples = selected_samples
        self.variable_index = variable_index
        self._in_recursion = False

    def __eq__(self, other):
        if self._in_recursion:
            return other.name == self.name
        return (
            super().__eq__(other)
            and isinstance(other, AncestralPlate)
            and self.variable_index == other.variable_index
        )

    def on_collecting_args(self, plates: [storch.Plate]) -> bool:
        """
        Filter the collected plates to only keep the AncestralPlates (with the same name) that has the highest variable index.
        :param plates:
        :return:
        """
        for plate in plates:
            if plate.name == self.name:
                if not isinstance(plate, AncestralPlate):
                    raise ValueError(
                        "Received a plate with name "
                        + plate.name
                        + " that is not also an AncestralPlate."
                    )
                if plate.variable_index > self.variable_index:
                    # Only keep ancestral plates with the highest variable index
                    return False
        return True

    def on_unwrap_tensor(self, tensor: storch.Tensor) -> storch.Tensor:
        """
        Gets called whenever the given tensor is being unwrapped and unsqueezed for batch use.
        :param tensor: The input tensor that is being unwrapped
        :return: The tensor that will be unwrapped and unsqueezed in the future. Can be a modification of the input tensor.
        """
        if self._in_recursion:
            # Required when calling storch.gather in this method. It will call on_unwrap_tensor again.
            return tensor
        for i, plate in enumerate(tensor.multi_dim_plates()):
            if plate.name != self.name:
                continue
            assert isinstance(plate, AncestralPlate)
            if plate.variable_index == self.variable_index:
                return tensor
            # This is true by the filtering at on_collecting_args
            assert plate.variable_index < self.variable_index

            parent_plates = []
            current_plate = self
            while current_plate.variable_index != plate.variable_index:
                parent_plates.append(current_plate)
                current_plate = current_plate.parent_plate
            assert current_plate == plate
            for parent_plate in reversed(parent_plates):
                expanded_selected_samples = right_expand_as(
                    parent_plate.selected_samples, tensor
                )
                tensor = storch.gather(
                    tensor, parent_plate.name, expanded_selected_samples
                )
        return tensor


def log1mexp(a: torch.Tensor) -> torch.Tensor:
    """See appendix A of http://jmlr.org/papers/v21/19-985.html.
    Numerically stable implementation of log(1-exp(a))"""
    r = torch.zeros_like(a)
    c = -0.693
    r[a > c] = (-a[a > c].expm1()).log()
    r[a <= c] = (-a[a <= c].exp()).log1p()
    return r


def right_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(...,) + (None,) * diff].expand(
        (-1,) * tensor.ndim + expand_as.shape[tensor.ndim :]
    )


def stochastic_beam_search(
    distribution: Distribution,
    k: int,
    amt_plates: int,
    joint_log_probs: Optional[torch.Tensor],
    perturbed_log_probs: Optional[torch.Tensor],
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Sample k events from the distribution without replacement.

    Implements Ancestral Gumbel-Top-k sampling with m=k, known as Stochastic Beam Search: http://jmlr.org/papers/v21/19-985.html.
    This sample from the distribution k items sequentially over the independent dimensions without replacement
    :param distribution: The distribution to sample from
    :param k: The amount of
    :param plate_dim Set to None if no dimension yet
    :return:
    """

    # TODO: This code is mostly taken from Expect(). May need to refactor to join these operations
    if not distribution.has_enumerate_support:
        raise ValueError(
            "Can only perform stochastic beam search for distributions with enumerable support."
        )

    support: torch.Tensor = distribution.enumerate_support(expand=True)

    sizes = support.shape[
        amt_plates + 1 : len(support.shape) - len(distribution.event_shape)
    ]

    ranges = []
    for size in sizes:
        ranges.append(list(range(size)))

    support_non_expanded: torch.Tensor = distribution.enumerate_support(expand=False)
    d_log_probs = distribution.log_prob(support_non_expanded)

    # Sample independent tensors in sequence
    sampled_support_indices = support.new_zeros(
        size=(k,) + support.shape[1:-1], dtype=torch.long
    )
    amt_samples = 0
    sampled_parent_indices = None
    if joint_log_probs is not None:
        # TODO: Is it at shape[0]?
        amt_samples = joint_log_probs.shape[0]
        sampled_parent_indices = support.new_zeros(
            size=(k,) + joint_log_probs.shape[1:], dtype=torch.long
        )
        sampled_parent_indices[:amt_samples] = right_expand_as(
            torch.arange(amt_samples), joint_log_probs
        )

    # Iterate over the different (conditionally) independent samples being taken
    for indices in itertools.product(*ranges):
        # Log probabilities of the different options for this sample step
        yv_log_probs = d_log_probs[(slice(None),) * (amt_plates + 1) + indices]
        if joint_log_probs is None:
            joint_log_probs = yv_log_probs
            # First condition on max being 0:
            perturbed_log_probs = 0.0
            first_sample = True
        else:
            # Returns |D_yv| x k x ...
            joint_log_probs = joint_log_probs.unsqueeze(0) + yv_log_probs.unsqueeze(1)
            first_sample = False

        # Sample |D_yv| (x k) x ... Gumbel variables
        gumbel_d = Gumbel(loc=joint_log_probs, scale=1.0)
        G_yv = gumbel_d.rsample()

        # Condition the Gumbel samples on the maximum of previous samples
        Z = G_yv.max(0)[0]
        T = perturbed_log_probs
        vi = T - G_yv + log1mexp(G_yv - Z)
        cond_G_yv = T - vi.relu() - torch.nn.Softplus()(-vi.abs())

        if first_sample:
            # No parent has been sampled yet
            amt_samples = min(k, cond_G_yv.shape[0])
            # Compute top k over the conditional log probs
            perturbed_log_probs, arg_top = torch.topk(cond_G_yv, amt_samples, dim=0)
            joint_log_probs = joint_log_probs.gather(dim=0, index=arg_top)
            # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
            indexing = (slice(0, amt_samples),) + (slice(None),) * amt_plates + indices
            sampled_support_indices[indexing] = arg_top
        else:
            cond_G_yv = cond_G_yv.reshape((-1,) + cond_G_yv.shape[2:])
            prev_amt_samples = amt_samples
            amt_samples = min(k, cond_G_yv.shape[0])
            # Gather corresponding joint log probabilities
            perturbed_log_probs, arg_top = torch.topk(cond_G_yv, amt_samples, dim=0)
            joint_log_probs = joint_log_probs.reshape(
                (-1,) + joint_log_probs.shape[2:]
            ).gather(dim=0, index=arg_top)
            # Keep track of what parents were sampled for the arg top
            chosen_parents = arg_top.remainder(prev_amt_samples)
            sampled_support_indices = sampled_support_indices.gather(
                dim=0, index=right_expand_as(chosen_parents, sampled_support_indices)
            )
            if sampled_parent_indices is not None:
                sampled_parent_indices = sampled_parent_indices.gather(
                    dim=0, index=chosen_parents
                )
            chosen_samples = arg_top / prev_amt_samples
            # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
            indexing = (slice(0, amt_samples),) + (slice(None),) * amt_plates + indices
            sampled_support_indices[indexing] = chosen_samples

    sampled_support_indices = sampled_support_indices[:amt_samples]
    # if sampled_parent_indices is not None:
    #     print(
    #         "cat",
    #         torch.cat(
    #             [
    #                 sampled_support_indices[:, 0].squeeze().unsqueeze(0),
    #                 sampled_parent_indices[:, 0].unsqueeze(0),
    #             ],
    #             dim=0,
    #         ).T,
    #     )
    expanded_indices = right_expand_as(sampled_support_indices, support)
    samples = support.gather(dim=0, index=expanded_indices)
    return samples, joint_log_probs, perturbed_log_probs, sampled_parent_indices
