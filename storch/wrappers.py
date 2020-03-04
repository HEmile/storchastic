from __future__ import annotations

from typing import Union, Any, Tuple

import storch
import torch
from collections.abc import Iterable, Mapping
import warnings

_context_stochastic = False
_context_deterministic = 0
_stochastic_parents = []
_context_name = None
_context_amt_samples = 0
_plate_links = []

Plate = Tuple[str, int]

# def plate_in(plate: Plate, l: [Plate]):
#     # Not using "plate in plates" because __eq__ is overriden, which opens a deterministic context, which
#     # calls this method again, causing infinite recursion
#     for _plate in l:
#         if plate_equal(plate, _plate):
#             return True
#     return False
#
#
# def plate_equal(plate_1: Plate, plate_2: Plate):
#     # Only doing == check on strings, as storch.Tensor's cannot reduce to bool
#     return plate_1 is plate_2 or isinstance(plate_1, str) and isinstance(plate_2, str) and plate_1 == plate_2


def is_iterable(a: Any):
    return isinstance(a, Iterable) and not isinstance(a, torch.Tensor) and not isinstance(a, str)


def _collect_parents_and_plates(a: Any, parents: [storch.Tensor], plates: [Plate]):
    if isinstance(a, storch.Tensor):
        parents.append(a)
        for plate in a.batch_links:
            if plate not in plates:
                plates.append(plate)
    elif is_iterable(a):
        for _a in a:
            _collect_parents_and_plates(_a, parents, plates)
    elif isinstance(a, Mapping):
        for _a in a.values():
            _collect_parents_and_plates(_a, parents, plates)


def _unsqueeze_and_unwrap(a: Any, multi_dim_plates: [Plate]):
    if isinstance(a, storch.Tensor):
        tensor = a._tensor

        # It can be possible that the ordering of the plates does not align with the ordering of the inputs.
        # This part corrects this.
        amt_recognized = 0
        links: [Plate] = a.multi_dim_plates()
        for i, plate in enumerate(multi_dim_plates):
            if plate in a.batch_links:
                if plate != links[amt_recognized]:
                    # The plate is also in the tensor, but not in the ordering expected. So switch that ordering
                    j = links.index(plate)
                    tensor = tensor.transpose(j, amt_recognized)
                    links[amt_recognized], links[j] = links[j], links[amt_recognized]
                amt_recognized += 1

        for i, plate in enumerate(multi_dim_plates):
            if plate not in a.batch_links:
                tensor = tensor.unsqueeze(i)
        return tensor
    elif is_iterable(a):
        l = []
        for _a in a:
            l.append(_unsqueeze_and_unwrap(_a, multi_dim_plates))
        if isinstance(a, tuple):
            return tuple(l)
        return l
    elif isinstance(a, Mapping):
        d = {}
        for k, _a in a.items():
            d[k] = _unsqueeze_and_unwrap(_a, multi_dim_plates)
        return d
    else:
        return a


def _unwrap(*args, **kwargs):
    parents: [storch.Tensor] = []
    plates: [Plate] = []

    # Collect parent tensors and plates
    _collect_parents_and_plates(args, parents, plates)
    _collect_parents_and_plates(kwargs, parents, plates)

    storch.wrappers._plate_links = plates

    # Get the list of plates with size larger than 1 for the unsqueezing of tensors
    multi_dim_plates = []
    for plate_name, plate_n in plates:
        if plate_n > 1:
            multi_dim_plates.append((plate_name, plate_n))

    # Unsqueeze and align batched dimensions so that batching works easily.
    unsqueezed_args = []
    for t in args:
        unsqueezed_args.append(_unsqueeze_and_unwrap(t, multi_dim_plates))
    unsqueezed_kwargs = {}
    for k, v in kwargs.items():
        unsqueezed_kwargs[k] = _unsqueeze_and_unwrap(v, multi_dim_plates)
    return unsqueezed_args, unsqueezed_kwargs, parents, plates


def _process_deterministic(o: Any, parents: [storch.Tensor], plates: [Plate], is_cost: bool, name: str):
    if o is None:
        return
    if isinstance(o, storch.Tensor):
        if o.stochastic:
            raise RuntimeError("Creation of stochastic storch Tensor within deterministic context")
        # TODO: Does this require shape checking? Parent/Plate checking?
        return o
    if isinstance(o, torch.Tensor):  # Explicitly _not_ a storch.Tensor
        t = storch.DeterministicTensor(o, parents, plates, is_cost, name=name)
        if is_cost and t.event_shape != ():
            # TODO: Make sure the o.size() check takes into account the size of the sample.
            raise ValueError(
                "Event shapes (ie, non batched dimensions) of cost nodes have to be single floating point numbers. ")
        return t
    raise NotImplementedError("Handling of other types of return values is currently not implemented: ", o)


def _deterministic(fn, is_cost: bool):
    def wrapper(*args, **kwargs):
        if storch.wrappers._context_stochastic:
            # TODO check if we can re-add this
            raise NotImplementedError(
                "It is currently not allowed to open a deterministic context in a stochastic context")
        if storch.wrappers._context_deterministic > 0:
            if is_cost:
                raise RuntimeError("Cannot call storch.cost from within a deterministic context.")

            # TODO: This is currently uncommented and it will in fact unwrap. This was required because it was, eg,
            # possible to open a deterministic context, passing distributions with storch.Tensors as parameters,
            # then doing computations on these parameters. This is because these storch.Tensors will not be unwrapped
            # in the deterministic context as the unwrapping only considers lists.
            # # We are already in a deterministic context, no need to wrap or unwrap as only the outer dependencies matter
            # return fn(*args, **kwargs)

        args, kwargs, parents, plates = _unwrap(*args, **kwargs)

        if not parents and not is_cost:
            return fn(*args, **kwargs)
        storch.wrappers._context_deterministic += 1
        outputs = fn(*args, **kwargs)
        storch.wrappers._context_deterministic -= 1
        if is_iterable(outputs):
            n_outputs = []
            for o in outputs:
                n_outputs.append(_process_deterministic(o, parents, plates, is_cost, fn.__name__ + str(len(n_outputs))))
            outputs = n_outputs
        else:
            outputs = _process_deterministic(outputs, parents, plates, is_cost, fn.__name__)
        storch.wrappers._plate_links = []
        return outputs

    return wrapper


def deterministic(fn):
    return _deterministic(fn, False)


def cost(fn):
    return _deterministic(fn, True)


def _self_deterministic(fn, self: storch.Tensor):
    fn = deterministic(fn)
    def wrapper(*args, **kwargs):
        # Inserts the self object at the beginning of the passed arguments. In essence, it "fakes" the self reference.
        args = list(args)
        args.insert(0, self)
        return fn(*args, **kwargs)
    return wrapper


def _process_stochastic(output: torch.Tensor, parents: [storch.Tensor], plates: [Plate]):
    if isinstance(output, storch.Tensor):
        if not output.stochastic:
            # TODO: Calls _add_parents so something is going wrong here
            # The Tensor was created by calling @deterministic within a stochastic context.
            # This means that we have to conservatively assume it is dependent on the parents
            output._add_parents(storch.wrappers._stochastic_parents)
        return output
    if isinstance(output, torch.Tensor):
        t = storch.DeterministicTensor(output, parents, plates, False)
        return t
    else:
        raise TypeError("All outputs of functions wrapped in @storch.stochastic "
                        "should be Tensors. At " + str(output))


def stochastic(fn):
    """
    Applies `fn` to the `inputs`. `fn` should return one or multiple `storch.Tensor`s.
    `fn` should not call `storch.stochastic` or `storch.deterministic`. `inputs` can include `storch.Tensor`s.
    :param fn:
    :return:
    """

    def wrapper(*args, **kwargs):
        if storch.wrappers._context_stochastic or storch.wrappers._context_deterministic > 0:
            raise RuntimeError("Cannot call storch.stochastic from within a stochastic or deterministic context.")
        storch.wrappers._context_stochastic = True
        storch.wrappers._context_name = fn.__name__
        storch.wrappers._context_amt_samples = 0
        # Save the parents
        args, kwargs, parents, plates = _unwrap(*args, **kwargs)
        storch.wrappers._stochastic_parents = parents

        outputs = fn(*args, **kwargs)

        # Add parents to the outputs
        if is_iterable(outputs):
            processed_outputs = []
            for o in outputs:
                processed_outputs.append(_process_stochastic(o, parents, plates))
        else:
            processed_outputs = _process_stochastic(outputs, parents, plates)
        storch.wrappers._context_stochastic = False
        storch.wrappers._stochastic_parents = []
        storch.wrappers._context_name = None
        storch.wrappers._plate_links = []
        return processed_outputs

    return wrapper


def _exception_wrapper(fn):
    def wrapper(*args, **kwargs):
        for a in args:
            if isinstance(a, storch.Tensor):
                raise RuntimeError("It is not allowed to call this method using storch.Tensor, likely because it exposes its"
                                   "wrapped tensor to Python.")
        return fn(*args, **kwargs)
    return wrapper

def _unpack_wrapper(fn):
    def wrapper(*args, **kwargs):
        new_args = []
        for a in args:
            if isinstance(a, storch.Tensor):
                new_args.append(a._tensor)
            else:
                new_args.append(a)
        return fn(*tuple(new_args), **kwargs)
    return wrapper