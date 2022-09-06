from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import Union, List, Optional, Tuple, Callable, Iterable

from storch import Plate

from storch.typing import AnyTensor
from storch.sampling.method import SamplingMethod
from torch.distributions import Distribution
import storch
import torch


class AncestralPlate(storch.Plate):
    def __init__(
        self,
        name: str,
        n: int,
        parents: List[storch.Plate],
        variable_index: int,
        parent_plate: AncestralPlate,
        selected_samples: Optional[storch.Tensor],
        log_probs: Optional[storch.Tensor],
        weight: Optional[storch.Tensor] = None,
    ):
        super().__init__(name, n, parents, weight)
        assert (not parent_plate and variable_index == 0) or (
            parent_plate.n <= self.n and parent_plate.variable_index < variable_index
        )
        self.parent_plate = parent_plate
        self.selected_samples = selected_samples
        if isinstance(log_probs, storch.Tensor):
            self.log_probs = storch.Tensor(
                log_probs._tensor,
                [log_probs],
                log_probs.plates + [self]
            )
        self.variable_index = variable_index
        self._in_recursion = False
        self._override_equality = False

    def __eq__(self, other):
        if not isinstance(other, AncestralPlate):
            return False
        if self._override_equality or other._override_equality:
            return other.name == self.name
        return (
            super().__eq__(other)
            and self.variable_index == other.variable_index
        )

    def __repr__(self):
        return (
            "(Ancestral, " + self.variable_index.__repr__() + super().__repr__() + ")"
        )

    def on_collecting_args(self, plates: [storch.Plate]) -> bool:
        """
        Filter the collected plates to only keep the AncestralPlates (with the same name) that has the highest variable index.
        :param plates:
        :return:
        """
        # TODO: It seems like in_recursion controls _override_equality. Does it have other uses? Can we remove it and just
        #  use _override_equality?
        # TODO: in the expand in on_unwrap_tensor, _override_equality needs to be true but isn't.
        if self._in_recursion:
            self._override_equality = True
        if any(map(lambda plate: isinstance(plate, AncestralPlate) and plate._in_recursion, plates)) and not self._in_recursion:
            return False
        return super().on_collecting_args(plates)

    def on_duplicate_plate(self, plate: storch.Plate) -> bool:
        if not isinstance(plate, AncestralPlate):
            raise ValueError(
                "Received a plate with name "
                + plate.name
                + " that is not also an AncestralPlate."
            )
        return plate.variable_index > self.variable_index

    def on_unwrap_tensor(self, tensor: storch.Tensor) -> storch.Tensor:
        """
        Gets called whenever the given tensor is being unwrapped and unsqueezed for batch use.
        This method should not be called on tensors whose variable index is higher than this plates.

        selected_samples is used to choose from the parent plates what is the previous element in the sequence.
        This is for example used in sampling without replacement.
        If set to None, it is assumed the different sequences are indexed by the plate dimension.

        :param tensor: The input tensor that is being unwrapped
        :return: The tensor that will be unwrapped and unsqueezed in the future. Can be a modification of the input tensor.
        """
        if self._in_recursion:
            # Required when calling storch.gather in this method. It will call on_unwrap_tensor again.
            return tensor
        # Find the corresponding ancestral plate
        for i, plate in enumerate(tensor.plates):
            if plate.name != self.name:
                continue
            assert isinstance(plate, AncestralPlate)
            if plate.variable_index == self.variable_index:
                return tensor
            # This is true by the filtering at on_collecting_args
            assert plate.variable_index < self.variable_index

            if self.selected_samples is None:
                # If this sequence method does not explicitly select from previous samples, just return it with the current plate
                new_plates = tensor.plates.copy()
                new_plates[i] = self
                return storch.Tensor(tensor._tensor, [tensor], new_plates)

            downstream_plates = []
            current_plate = self

            # Collect the list of plates from the tensors variable index to this plates variable index
            while current_plate.variable_index != plate.variable_index:
                downstream_plates.append(current_plate)
                current_plate = current_plate.parent_plate
            assert current_plate == plate

            # Go over all parent plates and gather their respective choices.
            for parent_plate in reversed(downstream_plates):
                self._in_recursion = True
                expanded_selected_samples = expand_with_ignore_as(
                    parent_plate.selected_samples, tensor, self.name
                )
                self._override_equality = False
                # Gather what samples of the tensor are chosen by this plate (parent_plate)
                tensor = storch.gather(
                    tensor, parent_plate.name, expanded_selected_samples
                )
                self._in_recursion = False
                self._override_equality = False
            break
        return tensor

    def index_in(self, plates: List[Plate]) -> int:
        return list(map(lambda p: p.name, plates)).index(self.name)

    def is_in(self, plates: Iterable[Plate]) -> bool:
        return any(map(lambda p: p.name == self.name, plates))

class SequenceDecoding(SamplingMethod):
    """
    Methods for generating sequences of discrete random variables.
    Examples: Simple ancestral sampling with replacement, beam search, Stochastic beam search (sampling without replacement)
    """

    EPS = 1e-8

    def __init__(self, plate_name: str, k: int, eos: None):
        super().__init__(plate_name)
        self.k = k
        self.eos = eos
        self.reset()
        self.plate_first = True

    def reset(self):
        super().reset()
        # Cumulative log probabilities of the samples
        self.joint_log_probs = None
        # Chosen parent samples at the previous sample step
        self.parent_indexing = None
        # The index of the currently sampled variable
        self.variable_index = 0
        # The previously created plates
        self.new_plate = None
        # What runs are already finished
        self.finished_samples = None
        # The sampled sequence so far
        self.seq = []

    def sample(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        orig_distr_plates: [storch.Plate],
        requires_grad: bool,
    ) -> (torch.Tensor, storch.Plate):

        """
        Sample from the distribution given the sequence so far.
        :param distribution: The distribution to sample from
        :return:
        """

        # Do the decoding step given the prepared tensors
        samples, new_joint_log_probs, self.parent_indexing = self.decode(
            distr, self.joint_log_probs, parents, orig_distr_plates
        )
        if new_joint_log_probs is not None:
            self.joint_log_probs = new_joint_log_probs
        else:
            # We'll assign it later
            self.joint_log_probs = None

        # Find out what sequences have reached the EOS token, and make sure to always sample EOS after that.
        # Does not contain the ancestral plate as this uses samples instead of s_tensor.
        if self.eos:
            self.finished_samples = samples.eq(self.eos)

        if isinstance(samples, storch.Tensor):
            k_index = samples.plate_dims
            plates = samples.plates
            samples = samples._tensor
        else:
            plate_names = list(map(lambda p: p.name, orig_distr_plates))
            k_index = plate_names.index(self.plate_name) if self.plate_name in plate_names else 0
            plates = orig_distr_plates


        plate_size = samples.shape[k_index]

        # Remove the ancestral plate, if it already happens to be in
        to_remove = None
        for plate in plates:
            if plate.name == self.plate_name:
                to_remove = plate
                break
        if to_remove:
            plates.remove(to_remove)

        # Create the newly updated plate
        self.new_plate = self.create_plate(plate_size, plates.copy())
        # TODO: This can probably be simplified by assuming plate_first
        plates.insert(k_index, self.new_plate)

        if self.parent_indexing is not None:
            assert isinstance(self.parent_indexing, storch.Tensor)
            # TODO: This too
            self.parent_indexing.plates.insert(k_index, self.new_plate)

        # Adjust the sequence based on parent samples chosen
        self.seq = list(map(lambda t: self.new_plate.on_unwrap_tensor(t), self.seq))

        # Construct the stochastic tensor
        s_tensor = storch.StochasticTensor(
            samples,
            parents,
            plates,
            self.plate_name,
            plate_size,
            distr,
            requires_grad or self.joint_log_probs is not None and self.joint_log_probs.requires_grad,
        )

        # Update joint log probs if decoding method did not compute it
        if new_joint_log_probs is None:
            s_log_probs = distr.log_prob(s_tensor)
            if self.joint_log_probs is not None:
                self.joint_log_probs += s_log_probs
            else:
                self.joint_log_probs = s_log_probs

            if self.finished_samples:
                # Make sure we do not change the log probabilities for samples that were already finished
                self.joint_log_probs = (
                        self.joint_log_probs * self.finished_samples
                        + self.joint_log_probs * (1 - self.finished_samples)
                )
            self.new_plate.log_probs = self.joint_log_probs
            s_tensor._requires_grad = s_tensor.requires_grad or self.joint_log_probs.requires_grad
        self.seq.append(s_tensor)

        # Increase variable index for next samples in sequence
        self.variable_index += 1
        return s_tensor, self.new_plate

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: storch.Plate
    ) -> Optional[storch.Tensor]:
        if self.eos:
            active = 1 - self.finished_samples
            amt_active: storch.Tensor = storch.sum(active, plate)
            return active / amt_active
        return super().weighting_function(tensor, plate)

    @abstractmethod
    def decode(
        self,
        distribution: Distribution,
        joint_log_probs: Optional[storch.Tensor],
        parents: [storch.Tensor],
        orig_distr_plates: [storch.Plate],
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        """
        Decode given the input arguments
        :param distribution: The distribution to decode
        :param joint_log_probs: The log probabilities of the samples so far. prev_plates x amt_samples
        :param parents: List of parents of this tensor
        :param orig_distr_plates: List of plates from the distribution. Can include the self plate k.
        :return: 3-tuple of `storch.Tensor`. 1: The sampled value. 2: The new joint log probabilities of the samples.
        3: How the samples index the parent samples. Can just be a range if there is no choosing happening.
        For all of these, the last plate index should be the plate index, with the other plates like `all_plates`
        """
        pass

    def create_plate(self, plate_size: int, plates: [storch.Plate]) -> AncestralPlate:
        return AncestralPlate(
            self.plate_name,
            plate_size,
            plates.copy(),
            self.variable_index,
            self.new_plate,
            self.parent_indexing,
            self.joint_log_probs,
            None,
        )

    def get_sampled_seq(self, finished: bool = False) -> [storch.StochasticTensor]:
        # TODO: the finished one doesn't return proper tensors.
        if finished:
            return list(map(lambda t: t[self.finished_samples], self.seq))
        return torch.cat(self.seq, dim=self.seq[0].plate_dims)

    def get_amt_finished(self) -> AnyTensor:
        if not self.eos:
            raise RuntimeError(
                "Cannot get the amount of finished sequences when eos is not set."
            )
        if self.finished_samples is None:
            return 0
        return torch.sum(self.finished_samples, -1)

    def get_unique_seqs(self):
        # TODO: Very experimental code
        seq_dim = self.seq[0].plate_dims
        cat_seq = torch.cat(self.seq, dim=seq_dim)
        return storch.unique(cat_seq, event_dim=0)

    def all_finished(self) -> bool:
        t = self.get_amt_finished().eq(self.k)
        # This is required because storch.Tensor's do not support .all() and .bool()
        if isinstance(t, storch.Tensor):
            return t._tensor.all().bool()
        return t.all().bool()


class MCDecoder(SequenceDecoding):

    def decode(
        self,
        distribution: Distribution,
        joint_log_probs: Optional[storch.Tensor],
        parents: [storch.Tensor],
        orig_distr_plates: [storch.Plate],
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):

        is_conditional_sample = False

        for plate in orig_distr_plates:
            if plate.name == self.plate_name:
                is_conditional_sample = True

        with storch.ignore_wrapping():
            if is_conditional_sample:
                sample = self.mc_sample(
                    distribution, parents, orig_distr_plates, 1
                ).squeeze(0)
            else:
                sample = self.mc_sample(distribution, parents, orig_distr_plates, self.k)

        if self.finished_samples:
            sample[self.finished_samples] = self.eos

        return sample, None, None





class IterDecoding(SequenceDecoding):

    def __init__(self, plate_name, k, eos):
        super().__init__(plate_name, k, eos)
        self.plate_first = False

    # Decodes a multivariable discrete distribution (independent over dimensions)
    def decode(
        self,
        distr: Distribution,
        joint_log_probs: Optional[storch.Tensor],
        parents: [storch.Tensor],
        orig_distr_plates: [storch.Plate],
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        """
        Decode given the input arguments
        :param distribution: The distribution to decode
        :param joint_log_probs: The log probabilities of the samples so far. prev_plates x amt_samples
        :param parents: List of parents of this tensor
        :param orig_distr_plates: List of plates from the distribution. Can include the self plate k.
        :return: 3-tuple of `storch.Tensor`. 1: The sampled value. 2: The new joint log probabilities of the samples.
        3: How the samples index the parent samples. Can just be None if there is no choosing happening.
        """
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

        # LEGEND FOR SHAPE COMMENTS
        # =========================
        # To make this code generalize to every bayesian network, complicated shape management is necessary.
        # The following are references to the shapes that are used within the method
        #
        # distr_plates: refers to the plates on the parameters of the distribution. Does *not* include
        #  the k? ancestral plate (possibly empty)
        # orig_distr_plates: refers to the plates on the parameters of the distribution, and *can* include
        #  the k? ancestral plate (possibly empty)
        # prev_plates: refers to the plates of the previous sampled variable in this swr sample (possibly empty)
        # plates: refers to all plates except this ancestral plate, of which there are amt_plates. The first plates are
        #  the distr_plates, after that the prev_plates that are _not_ in distr_plates.
        #  It is composed of distr_plate x (ancstr_plates - distr_plates)
        # events: refers to the conditionally independent dimensions of the distribution (the distributions batch shape minus the plates)
        # k: refers to self.k
        # k?: refers to an optional plate dimension of this ancestral plate. It either doesn't exist, or is the sample
        #  dimension. If it exists, this means this sample is conditionally dependent on ancestors.
        # |D_yv|: refers to the *size* of the domain
        # amt_samples: refers to the current amount of sampled sequences. amt_samples <= k, but it can be lower if there
        #  are not enough events to sample from (eg |D_yv| < k)
        # event_shape: refers to the *shape* of the domain elements
        #  (can be 0, eg Categorical, or equal to |D_yv| for OneHotCategorical)
        ancestral_distrplate_index = -1
        is_conditional_sample = False

        multi_dim_distr_plates = []
        multi_dim_index = 0
        for plate in orig_distr_plates:
            if plate.n > 1:
                if plate.name == self.plate_name:
                    ancestral_distrplate_index = multi_dim_index
                    is_conditional_sample = True
                else:
                    multi_dim_distr_plates.append(plate)
                multi_dim_index += 1
        # plates? x k x events x

        # TODO: This doesn't properly combine two ancestral plates with the same name but different variable index
        #  (they should merge).
        all_multi_dim_plates = multi_dim_distr_plates.copy()
        if self.variable_index > 0:
            # Previous variables have been sampled. add the prev_plates to all_plates
            for plate in self.joint_log_probs.multi_dim_plates():
                if plate not in multi_dim_distr_plates:
                    all_multi_dim_plates.append(plate)

        amt_multi_dim_plates = len(all_multi_dim_plates)
        amt_multi_dim_distr_plates = len(multi_dim_distr_plates)
        amt_multi_dim_orig_distr_plates = amt_multi_dim_distr_plates + (
            1 if is_conditional_sample else 0
        )
        amt_multi_dim_prev_plates = amt_multi_dim_plates - amt_multi_dim_distr_plates
        if not distr.has_enumerate_support:
            raise ValueError("Can only decode distributions with enumerable support.")

        with storch.ignore_wrapping():
            # |D_yv| x (|distr_plates| + |k?| + |event_dims|) * (1,) x |D_yv|
            support_non_expanded: torch.Tensor = distr.enumerate_support(expand=False)
            # Compute the log-probability of the different events
            # |D_yv| x distr_plate[0] x ... k? ... x distr_plate[n-1] x events
            d_log_probs = distr.log_prob(support_non_expanded)

            # Note: Use amt_orig_distr_plates here because it might include k? dimension. amt_distr_plates filters this one.
            # distr_plate[0] x ... k? ... x distr_plate[n-1] x |D_yv| x events
            d_log_probs = storch.Tensor(
                d_log_probs.permute(
                    tuple(range(1, amt_multi_dim_orig_distr_plates + 1))
                    + (0,)
                    + tuple(
                        range(
                            amt_multi_dim_orig_distr_plates + 1, len(d_log_probs.shape)
                        )
                    )
                ),
                [],
                orig_distr_plates,
            )

        # |D_yv| x distr_plate[0] x ... x k? x ... x distr_plate[n-1] x events x event_shape
        support = distr.enumerate_support(expand=True)

        if is_conditional_sample:
            # Reduce ancestral dimension in the support. As the dimension is just an expanded version, this should
            # not change the underlying data.
            # |D_yv| x distr_plates x events x event_shape
            support = support[(slice(None),) * (ancestral_distrplate_index + 1) + (0,)]

            # Gather the correct log probabilities
            # distr_plate[0] x ... k ... x distr_plate[n-1] x |D_yv| x events
            # TODO: Move this down below to the other scary TODO
            d_log_probs = self.new_plate.on_unwrap_tensor(d_log_probs)
            # Permute the dimensions of d_log_probs st the k dimension is after the plates.
            for i, plate in enumerate(d_log_probs.multi_dim_plates()):
                if plate.name == self.plate_name:
                    d_log_probs.plates.remove(plate)
                    # distr_plates x k x |D_yv| x events
                    d_log_probs._tensor = d_log_probs._tensor.permute(
                        tuple(range(0, i))
                        + tuple(range(i + 1, amt_multi_dim_orig_distr_plates))
                        + (i,)
                        + tuple(
                            range(
                                amt_multi_dim_orig_distr_plates, len(d_log_probs.shape)
                            )
                        )
                    )
                    break

        # Equal to event_shape
        element_shape = distr.event_shape
        support_permutation = (
            tuple(range(1, amt_multi_dim_distr_plates + 1))
            + (0,)
            + tuple(range(amt_multi_dim_distr_plates + 1, len(support.shape)))
        )
        # distr_plates x |D_yv| x events x event_shape
        support = support.permute(support_permutation)

        if amt_multi_dim_plates != amt_multi_dim_distr_plates:
            # If previous samples had plate dimensions that are not in the distribution plates, add these to the support.
            support = support[
                (slice(None),) * amt_multi_dim_distr_plates
                + (None,) * amt_multi_dim_prev_plates
            ]
            all_plate_dims = tuple(map(lambda _p: _p.n, all_multi_dim_plates))
            # plates x |D_yv| x events x event_shape (where plates = distr_plates x prev_plates)
            support = support.expand(
                all_plate_dims + (-1,) * (len(support.shape) - amt_multi_dim_plates)
            )
        # plates x |D_yv| x events x event_shape
        support = storch.Tensor(support, [], all_multi_dim_plates)

        # Equal to events: Shape for the different conditional independent dimensions
        event_shape = support.shape[
            amt_multi_dim_plates + 1 : -len(element_shape)
            if len(element_shape) > 0
            else None
        ]

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
                size=support.shape[:amt_multi_dim_plates] + (self.k,), dtype=torch.long
            )

            # probably can go wrong if plates are missing.
            parent_indexing[..., :amt_samples] = left_expand_as(
                torch.arange(amt_samples), parent_indexing
            )
        # plates x k x events
        sampled_support_indices = support.new_zeros(
            size=support.shape[:amt_multi_dim_plates]  # plates
            + (self.k,)
            + support.shape[
                amt_multi_dim_plates + 1 : -len(element_shape)
                if len(element_shape) > 0
                else None
            ],  # events
            dtype=torch.long,
        )
        # Sample independent tensors in sequence
        # Iterate over the different (conditionally) independent samples being taken (the events)
        for indices in itertools.product(*ranges):
            # Log probabilities of the different options for this sample step (event)
            # distr_plates x k? x |D_yv|
            yv_log_probs = d_log_probs[(...,) + indices]
            (
                sampled_support_indices,
                joint_log_probs,
                parent_indexing,
                amt_samples,
            ) = self.decode_step(
                indices,
                yv_log_probs,
                joint_log_probs,
                sampled_support_indices,
                parent_indexing,
                is_conditional_sample,
                amt_multi_dim_plates,
                amt_samples,
            )
        # Finally, index the support using the sampled indices to get the sample!
        if amt_samples < self.k:
            # plates x amt_samples x events
            sampled_support_indices = sampled_support_indices[
                (...,) + (slice(amt_samples),)
            ]
        expanded_indices = right_expand_as(sampled_support_indices, support)
        sample = support.gather(dim=amt_multi_dim_plates, index=expanded_indices)
        return sample, joint_log_probs, parent_indexing

    @abstractmethod
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
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor, int):
        """
        Decode given the input arguments for a specific event
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
        pass


def expand_with_ignore_as(
    tensor, expand_as, ignore_dim: Union[str, int]
) -> torch.Tensor:
    """
    Expands the tensor like expand_as, but ignores a single dimension.
    Ie, if tensor is of size a x b,  expand_as of size d x a x c and dim=-1, then the return will be of size d x a x b.
    It also automatically expands all plate dimensions correctly.
    :param ignore_dim: Can be a string referring to the plate dimension
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
        return storch.deterministic(
            _expand_with_ignore, expand_plates=True, dim=ignore_dim
        )(tensor, expand_as)
    return storch.deterministic(_expand_with_ignore, expand_plates=True)(
        tensor, expand_as, ignore_dim
    )


@storch.deterministic(l_broadcast=False)
def right_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(...,) + (None,) * diff].expand(
        (-1,) * tensor.ndim + expand_as.shape[tensor.ndim :]
    )


def left_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(None,) * diff].expand(expand_as.shape[:diff] + (-1,) * tensor.ndim)
