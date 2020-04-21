from __future__ import annotations
from typing import Optional, Union

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
        # Cumulative log probabilities of the samples
        self.joint_log_probs = None
        # Cumulative perturbed log probabilities of the samples
        self.perturbed_log_probs = None
        # Chosen parent samples at the previous sample step
        self.parent_indexing = None
        # The index of the currently sampled variable
        self.variable_index = 0
        # The previously created plates
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
        samples = self.stochastic_beam_search(distr, plates)
        print(samples)
        self.variable_index += 1
        return samples

    def stochastic_beam_search(
        self, distribution: Distribution, plates: [storch.Plate],
    ) -> torch.Tensor:
        """
        Sample k events from the distribution without replacement.

        Implements Ancestral Gumbel-Top-k sampling with m=k, known as Stochastic Beam Search: http://jmlr.org/papers/v21/19-985.html.
        This sample from the distribution k items sequentially over the independent dimensions without replacement
        :param distribution: The distribution to sample from
        :return:
        """

        # plates: refers to all plates except this ancestral plate, of which there are amt_plates
        # events: refers to the conditionally independent dimensions of the distribution (the distributions batch shape minus the plates)
        # k: refers to self.k
        # k?: refers to an optional plate dimension of this ancestral plate. It either doesn't exist, or is the sample
        #  dimension. If it exists, this means this sample is conditionally dependent on earlier samples.
        # |D_yv|: refers to the *size* of the domain
        # event_shape: refers to the *shape* of the domain elements (can be 0, eg Categorical, or equal to |D_yv| for OneHotCategorical)

        # This code has three parts.
        # The first prepares all necessary tensors to make sure they can be easily indexed.
        # This part is quite long as there are many cases.
        # 1) There have not been any variables sampled so far.
        # 2) There have been variables sampled, but their results are NOT used to compute the input distribution.
        #    in other words, the variable to sample is independent of the other sampled variables. However,
        #    we should still keep track of the other sampled variables to make sure that it still samples without
        #    replacement properly. In this case, the ancestral plate is not in the plates attribute.
        #    We also have to make sure that we know in the future what samples are chosen for the _other_ samples.
        # 3) There have been parents sampled, and this variable is dependent on at least some of them.
        #    The plates list then contains the ancestral plate. We need to make sure we compute the joint log probs
        #    for the conditional samples (ie, based on the different sampled variables in the ancestral dimension).
        # The second part is a loop over all options for the event dimensions. This samples these conditionally
        # independent samples in sequence. It samples indexes, not events.
        # The third part after the loop uses the sampled indexes and matches it to the events to be used.

        ancestral_plate_index = -1
        is_conditional_sample = False
        for i, plate in enumerate(plates):
            if plate.name == self.plate_name:
                ancestral_plate_index = i
                is_conditional_sample = True
                break

        amt_plates = len(plates) - (1 if is_conditional_sample else 0)
        if not distribution.has_enumerate_support:
            raise ValueError(
                "Can only perform stochastic beam search for distributions with enumerable support."
            )

        # |D_yv| x plate[0] x ... x k? x ... x plate[n-1] x events x event_shape
        # TODO: This won't work if k is already in the plates. I think I'd have to remove that dimension?
        support = distribution.enumerate_support(expand=True)

        if ancestral_plate_index != -1:
            # Reduce ancestral dimension in the support. As the dimension is just an expanded version, this should
            # not change the underlying data.
            support = support[
                (..., 0)
                + (slice(None),) * (len(support.shape) - ancestral_plate_index - 1)
            ]

        # Equal to event_shape
        element_shape = distribution.event_shape
        support_permutation = (
            tuple(range(1, amt_plates + 1))
            + (0,)
            + tuple(range(amt_plates + 1, len(support.shape)))
        )
        # plate[0] x ... x k? x ... x plate[n-1] x |D_yv| x events x event_shape
        support = support.permute(support_permutation)
        support_k_index = amt_plates

        # Shape for the different conditional independent dimensions
        events = support.shape[amt_plates + 1 : -len(element_shape)]

        ranges = []
        for size in events:
            ranges.append(list(range(size)))

        with storch.ignore_wrapping():
            # |D_yv| x (amt_plates + |k?| + |event_dims|) * (1,) x |D_yv|
            support_non_expanded: torch.Tensor = distribution.enumerate_support(
                expand=False
            )
            # |D_yv| x plate[0] x ... k? ... x plate[n-1] x events
            d_log_probs = distribution.log_prob(support_non_expanded)
            # Note: Use len(plates) here because it might include k? dimension. amt_plates filter this one.
            # plate[0] x ... k? ... x plate[n-1] x |D_yv| x events
            d_log_probs = storch.Tensor(
                d_log_probs.permute(
                    tuple(range(1, len(plates) + 1))
                    + (0,)
                    + tuple(range(len(plates) + 1, len(plates) + 1 + len(events)))
                ),
                [],
                plates,
            )
        if is_conditional_sample:
            # Gather the correct samples
            # plate[0] x ... k ... x plate[n-1] x |D_yv| x events
            d_log_probs = self.last_plate.on_unwrap_tensor(d_log_probs)
            # Permute the dimensions of d_log_probs st the k dimension is the rightmost event dimension.
            for i, plate in enumerate(d_log_probs.multi_dim_plates()):
                if plate.name == self.plate_name:
                    # k is present in the plates
                    d_log_probs.plates.remove(plate)
                    # plates x k x |D_yv| x events
                    d_log_probs._tensor = d_log_probs._tensor.permute(
                        tuple(range(0, i))
                        + tuple(range(i + 1, len(plates)))
                        + (i,)
                        + tuple(range(len(plates), len(d_log_probs.shape)))
                    )
                    break

        # plates x events x k
        sampled_support_indices = support.new_zeros(
            size=support.shape[:support_k_index]  # plates
            + (self.k,)
            + support.shape[support_k_index + 1 : -len(element_shape)],  # events
            dtype=torch.long,
        )
        amt_samples = 0
        self.parent_indexing = None
        if self.joint_log_probs is not None:
            # self.joint_log_probs: plates x amt_samples
            # TODO: Is it at shape[-1]?
            amt_samples = self.joint_log_probs.shape[-1]
            self.parent_indexing = support.new_zeros(
                size=self.joint_log_probs.shape[:-1] + (self.k,), dtype=torch.long
            )
            # probably can go wrong if plates are missing.
            self.parent_indexing[:amt_samples] = right_expand_as(
                torch.arange(amt_samples), self.joint_log_probs
            )

        # Sample independent tensors in sequence
        # Iterate over the different (conditionally) independent samples being taken (the events)
        for indices in itertools.product(*ranges):
            # Log probabilities of the different options for this sample step (event)
            # plates x k? x |D_yv|
            yv_log_probs = d_log_probs[(...,) + indices]
            if self.joint_log_probs is None:
                # We also know that k? is not present, so plates x |D_yv|
                all_joint_log_probs = yv_log_probs
                # First condition on max being 0:
                self.perturbed_log_probs = 0.0
                first_sample = True
            elif is_conditional_sample:
                # TODO: Are the indexes of d_log_probs still going to be correct when different parents are chosen?
                #  PRETTY SURE THEY AREN'T. Ie, we'd need to do the gathering step in lines 145-160 at every loop
                # self.joint_log_probs: plates x k
                # plates x k x |D_yv|
                all_joint_log_probs = self.joint_log_probs.unsqueeze(-1) + yv_log_probs
            else:
                # self.joint_log_probs: plates x k
                # plates x k x |D_yv|
                all_joint_log_probs = self.joint_log_probs.unsqueeze(
                    -1
                ) + yv_log_probs.unsqueeze(-2)
                first_sample = False

            # Sample plates x k? x |D_yv| Gumbel variables
            gumbel_d = Gumbel(loc=all_joint_log_probs, scale=1.0)
            G_yv = gumbel_d.rsample()

            # Condition the Gumbel samples on the maximum of previous samples
            # plates x k
            Z = G_yv.max(dim=-1)[0]
            T = self.perturbed_log_probs
            vi = T - G_yv + log1mexp(G_yv - Z)
            # plates (x k) x |D_yv|
            cond_G_yv = T - vi.relu() - torch.nn.Softplus()(-vi.abs())

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
                self.joint_log_probs = all_joint_log_probs.gather(dim=-1, index=arg_top)
                # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
                # None * amt_plates + (indices for events) + amt_samples
                indexing = (
                    (slice(None),) * amt_plates + (slice(0, amt_samples),) + indices
                )
                sampled_support_indices[indexing] = arg_top
            else:
                # plates x (k * |D_yv|) (k == prev_amt_samples, in this case)
                cond_G_yv = cond_G_yv.reshape(cond_G_yv.shape[:-2] + (-1,))
                prev_amt_samples = amt_samples
                # We can sample at most the amount of what we previous sampled, combined with every option in the current domain
                # That is: prev_amt_samples * |D_yv|. But we also want to limit by k.
                amt_samples = min(self.k, cond_G_yv.shape[-1])
                # Take the top k over conditional perturbed log probs
                # plates x amt_samples
                self.perturbed_log_probs, arg_top = torch.topk(
                    cond_G_yv, amt_samples, dim=-1
                )
                # Gather corresponding joint log probabilities. First reshape like previous to plates x (k * |D_yv|).
                self.joint_log_probs = all_joint_log_probs.reshape(
                    cond_G_yv.shape[:-2] + (-1,)
                ).gather(dim=-1, index=arg_top)

                # |D_yv|
                size_domain = yv_log_probs.shape[-1]

                # Keep track of what parents were sampled for the arg top
                chosen_parents = right_expand_as(
                    arg_top / size_domain, sampled_support_indices,
                )
                sampled_support_indices = sampled_support_indices.gather(
                    dim=support_k_index, index=chosen_parents,
                )
                if self.parent_indexing is not None:
                    self.parent_indexing = self.parent_indexing.gather(
                        dim=-1, index=chosen_parents
                    )
                chosen_samples = arg_top.remainder(size_domain)
                # Index for the selected samples. Uses slice(amt_samples) for the first index in case k > |D_yv|
                indexing = (
                    (slice(None),) * amt_plates + (slice(0, amt_samples),) + indices
                )
                sampled_support_indices[indexing] = chosen_samples

        if amt_samples < self.k:
            # plates x amt_samples x events
            sampled_support_indices = sampled_support_indices[
                (...,) + (slice(amt_samples),) + (slice(None),) * len(events)
            ]
        expanded_indices = right_expand_as(sampled_support_indices, support)
        return support.gather(dim=support_k_index, index=expanded_indices)

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
            self.parent_indexing,
            None,
        )
        if self.parent_indexing is not None:
            self.parent_indexing.plates.append(self.last_plate)
        return self.last_plate

    def on_plate_already_present(self, plate: storch.Plate):
        if (
            not isinstance(plate, AncestralPlate)
            or plate.variable_index > self.variable_index
            or plate.n > self.k
        ):
            super().on_plate_already_present(plate)


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
        self._override_equality = False

    def __eq__(self, other):
        if self._override_equality:
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
        if self._in_recursion:
            self._override_equality = True
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
                self._in_recursion = True
                expanded_selected_samples = expand_with_ignore_as(
                    self.selected_samples, tensor, self.name
                )
                self._override_equality = False
                tensor = storch.gather(
                    tensor, parent_plate.name, expanded_selected_samples
                )
                self._in_recursion = False
                self._override_equality = False
            break
        return tensor


def log1mexp(a: torch.Tensor) -> torch.Tensor:
    """See appendix A of http://jmlr.org/papers/v21/19-985.html.
    Numerically stable implementation of log(1-exp(a))"""
    r = torch.zeros_like(a)
    c = -0.693
    assert (a._tensor <= 0).all()
    _b = -a[a > c].expm1()
    r[a > c] = (-a[a > c].expm1()).log()
    r[a <= c] = (-a[a <= c].exp()).log1p()
    return r


def right_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(...,) + (None,) * diff].expand(
        (-1,) * tensor.ndim + expand_as.shape[tensor.ndim :]
    )


def left_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(None,) * diff].expand(expand_as.shape[:diff] + (-1,) * tensor.ndim)


def expand_with_ignore_as(
    tensor, expand_as, ignore_dim: Union[str, int]
) -> torch.Tensor:
    """
    Expands the tensor like expand_as, but ignores a single dimension.
    Ie, if tensor is of size a x b,  expand_as of size d x a x c and dim=-1, then the return will be of size d x a x b
    :param ignore_dim: Can be a string referring to the plate dimension
    TODO: This is pretty broken because of how extra dimensions are added. It assumes all extra dimensions are
     right of the non-singleton dimensions.
    """
    # diff = expand_as.ndim - tensor.ndim
    def _expand_with_ignore(tensor, expand_as, dim: int):
        new_dims = expand_as.ndim - tensor.ndim
        # after_dims = tensor.ndim - dim
        return tensor[(...,) + (None,) * new_dims].expand(
            expand_as.shape[:dim]
            + (-1,)
            + (expand_as.shape[dim + 1 :] if dim != -1 else ())
        )

    if isinstance(ignore_dim, str):
        return storch.deterministic(_expand_with_ignore, dim=ignore_dim)(
            tensor, expand_as
        )
    return storch.deterministic(_expand_with_ignore)(tensor, expand_as, ignore_dim)
