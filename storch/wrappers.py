from __future__ import annotations

from typing import Union, Any, Tuple, List, Optional

import storch
import torch
from collections.abc import Iterable, Mapping
from functools import wraps
from storch.exceptions import IllegalStorchExposeError

_context_stochastic = False
_context_deterministic = 0
_stochastic_parents = []
_context_name = None
_plate_links = []
_ignore_wrap = False

# TODO: This is_iterable thing is a bit annoying: We really only want to unwrap them if they contain storch
#  Tensors, and then only for some types. Should rethink, maybe. Is unwrapping even necessary if the base torch methods
#  are all overriden? Maybe, see torch.cat?
def is_iterable(a: Any):
    return (
        isinstance(a, Iterable)
        and not isinstance(a, torch.Tensor)
        and not isinstance(a, str)
        and not isinstance(a, torch.Storage)
    )


def _collect_parents_and_plates(
    a: Any, parents: [storch.Tensor], plates: [storch.Plate]
) -> int:
    if isinstance(a, storch.Tensor):
        parents.append(a)
        for plate in a.plates:
            if plate not in plates:
                plates.append(plate)
        return a.event_dims
    elif is_iterable(a):
        max_event_dim = 0
        for _a in a:
            max_event_dim = max(
                max_event_dim, _collect_parents_and_plates(_a, parents, plates)
            )
        return max_event_dim
    elif isinstance(a, Mapping):
        max_event_dim = 0
        for _a in a.values():
            max_event_dim = max(
                max_event_dim, _collect_parents_and_plates(_a, parents, plates)
            )
        return max_event_dim
    return 0


def _unsqueeze_and_unwrap(
    a: Any, multi_dim_plates: [storch.Plate], align_tensors: bool, event_dims: int
):
    if isinstance(a, storch.Tensor):
        tensor = a._tensor
        if not align_tensors:
            return tensor
        # Automatically **RIGHT** broadcast. Ensure each tensor has an equal amount of event dims by inserting dimensions to the right
        # TODO: What do we think about this design?
        if a.event_dims < event_dims:
            tensor = tensor[(...,) + (None,) * (event_dims - a.event_dims)]

        # It can be possible that the ordering of the plates does not align with the ordering of the inputs.
        # This part corrects this.
        amt_recognized = 0
        links: [storch.Plate] = list(a.multi_dim_plates())
        for i, plate in enumerate(multi_dim_plates):
            if plate in a.plates:
                if plate != links[amt_recognized]:
                    # The plate is also in the tensor, but not in the ordering expected. So switch that ordering
                    j = links.index(plate)
                    tensor = tensor.transpose(j, amt_recognized)
                    links[amt_recognized], links[j] = links[j], links[amt_recognized]
                amt_recognized += 1

        for i, plate in enumerate(multi_dim_plates):
            if plate not in a.plates:
                tensor = tensor.unsqueeze(i)
        return tensor
    elif is_iterable(a):
        l = []
        for _a in a:
            l.append(
                _unsqueeze_and_unwrap(_a, multi_dim_plates, align_tensors, event_dims)
            )
        if isinstance(a, tuple):
            return tuple(l)
        return l
    elif isinstance(a, Mapping):
        d = {}
        for k, _a in a.items():
            d[k] = _unsqueeze_and_unwrap(
                _a, multi_dim_plates, align_tensors, event_dims
            )
        return d
    else:
        return a


def _handle_args(unwrap=True, align_tensors=True, *args, **kwargs):
    parents: [storch.Tensor] = []
    plates: [storch.Plate] = []
    max_event_dim = max(
        # Collect parent tensors and plates
        _collect_parents_and_plates(args, parents, plates),
        _collect_parents_and_plates(kwargs, parents, plates),
    )
    # Get the list of plates with size larger than 1 for the unsqueezing of tensors
    multi_dim_plates = []
    for plate in plates:
        if plate.n > 1:
            multi_dim_plates.append(plate)
    if unwrap:
        # Unsqueeze and align batched dimensions so that batching works easily.
        unsqueezed_args = []
        for t in args:
            unsqueezed_args.append(
                _unsqueeze_and_unwrap(t, multi_dim_plates, align_tensors, max_event_dim)
            )
        unsqueezed_kwargs = {}
        for k, v in kwargs.items():
            unsqueezed_kwargs[k] = _unsqueeze_and_unwrap(
                v, multi_dim_plates, align_tensors, max_event_dim
            )
        return unsqueezed_args, unsqueezed_kwargs, parents, plates
    return args, kwargs, parents, plates


def _process_deterministic(
    o: Any, parents: [storch.Tensor], plates: [storch.Plate], name: str
):
    if o is None:
        return
    if isinstance(o, storch.Tensor):
        if o.stochastic:
            raise RuntimeError(
                "Creation of stochastic storch Tensor within deterministic context"
            )
        # TODO: Does this require shape checking? Parent/Plate checking?
        return o
    if isinstance(o, torch.Tensor):  # Explicitly _not_ a storch.Tensor
        t = storch.Tensor(o, parents, plates, name=name)
        return t
    raise NotImplementedError(
        "Handling of other types of return values is currently not implemented: ", o
    )


def _deterministic(
    fn,
    *,
    unwrap: bool = True,
    align_tensors=True,
    reduce_plates: Optional[Union[str, List[str]]] = None
):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if storch.wrappers._context_stochastic:
            # TODO check if we can re-add this
            raise NotImplementedError(
                "It is currently not allowed to open a deterministic context in a stochastic context"
            )
            # if storch.wrappers._context_deterministic > 0:
            #     if is_cost:
            #         raise RuntimeError("Cannot call storch.cost from within a deterministic context.")

            # TODO: This is currently uncommented and it will in fact unwrap. This was required because it was, eg,
            # possible to open a deterministic context, passing distributions with storch.Tensors as parameters,
            # then doing computations on these parameters. This is because these storch.Tensors will not be unwrapped
            # in the deterministic context as the unwrapping only considers lists.
            # # We are already in a deterministic context, no need to wrap or unwrap as only the outer dependencies matter
            # return fn(*args, **kwargs)

        new_args, new_kwargs, parents, plates = _handle_args(
            unwrap, align_tensors, *args, **kwargs
        )

        if not parents:
            return fn(*args, **kwargs)
        args = new_args
        kwargs = new_kwargs

        storch.wrappers._context_deterministic += 1

        try:
            outputs = fn(*args, **kwargs)
        finally:
            storch.wrappers._context_deterministic -= 1

        if storch.wrappers._ignore_wrap:
            return outputs
        nonlocal reduce_plates
        if reduce_plates:
            if isinstance(reduce_plates, str):
                reduce_plates = [reduce_plates]
            plates = [p for p in plates if p.name not in reduce_plates]
        if is_iterable(outputs):
            n_outputs = []
            for o in outputs:
                n_outputs.append(
                    _process_deterministic(
                        o, parents, plates, fn.__name__ + str(len(n_outputs))
                    )
                )
            outputs = n_outputs
        else:
            outputs = _process_deterministic(outputs, parents, plates, fn.__name__)
        return outputs

    return wrapper


def deterministic(fn=None, *, unwrap=True, align_tensors=True):
    """
    Wraps the input function around a deterministic storch wrapper.
    This wrapper unwraps :class:`~storch.Tensor` objects to :class:`~torch.Tensor` objects, aligning the tensors
    according to the plates, then runs `fn` on the unwrapped Tensors.
    :param fn: Function to wrap.
    :param unwrap: Set to False to prevent unwrapping :classs:`~storch.Tensor` objects.
    :return: The wrapped function `fn`.
    """
    if fn:
        return _deterministic(fn, unwrap=unwrap, align_tensors=align_tensors)
    return lambda _f: _deterministic(_f, unwrap=unwrap, align_tensors=align_tensors)


def reduce(fn, plates: Union[str, List[str]]):
    """
        Wraps the input function around a deterministic storch wrapper.
        This wrapper unwraps :class:`~storch.Tensor` objects to :class:`~torch.Tensor` objects, aligning the tensors
        according to the plates, then runs `fn` on the unwrapped Tensors. It will reduce the plates given by `plates`.
        :param fn: Function to wrap.
        :param unwrap: Set to False to prevent unwrapping :classs:`~storch.Tensor` objects.
        :return: The wrapped function `fn`.
        """
    if storch._debug:
        print("Reducing plates", plates)
    return _deterministic(fn, reduce_plates=plates)


def _self_deterministic(fn, self: storch.Tensor):
    fn = deterministic(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Inserts the self object at the beginning of the passed arguments. In essence, it "fakes" the self reference.
        args = list(args)
        args.insert(0, self)
        return fn(*args, **kwargs)

    return wrapper


def _process_stochastic(
    output: torch.Tensor, parents: [storch.Tensor], plates: [storch.Plate]
):
    if isinstance(output, storch.Tensor):
        if not output.stochastic:
            # TODO: Calls _add_parents so something is going wrong here
            # The Tensor was created by calling @deterministic within a stochastic context.
            # This means that we have to conservatively assume it is dependent on the parents
            output._add_parents(storch.wrappers._stochastic_parents)
        return output
    if isinstance(output, torch.Tensor):
        t = storch.Tensor(output, parents, plates)
        return t
    else:
        raise TypeError(
            "All outputs of functions wrapped in @storch.stochastic "
            "should be Tensors. At " + str(output)
        )


def stochastic(fn):
    """
    Applies `fn` to the `inputs`. `fn` should return one or multiple `storch.Tensor`s.
    `fn` should not call `storch.stochastic` or `storch.deterministic`. `inputs` can include `storch.Tensor`s.
    :param fn:
    :return:
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if (
            storch.wrappers._context_stochastic
            or storch.wrappers._context_deterministic > 0
        ):
            raise RuntimeError(
                "Cannot call storch.stochastic from within a stochastic or deterministic context."
            )

        # Save the parents
        args, kwargs, parents, plates = _handle_args(*args, **kwargs)

        storch.wrappers._plate_links = plates
        storch.wrappers._stochastic_parents = parents
        storch.wrappers._context_stochastic = True
        storch.wrappers._context_name = fn.__name__

        try:
            outputs = fn(*args, **kwargs)
        finally:
            storch.wrappers._plate_links = []
            storch.wrappers._stochastic_parents = []
            storch.wrappers._context_stochastic = False
            storch.wrappers._context_name = None

        # Add parents to the outputs
        if is_iterable(outputs):
            processed_outputs = []
            for o in outputs:
                processed_outputs.append(_process_stochastic(o, parents, plates))
        else:
            processed_outputs = _process_stochastic(outputs, parents, plates)
        return processed_outputs

    return wrapper


def _exception_wrapper(fn):
    def wrapper(*args, **kwargs):
        for a in args:
            if isinstance(a, storch.Tensor):
                raise IllegalStorchExposeError(
                    "It is not allowed to call this method using storch.Tensor, likely "
                    "because it exposes its wrapped tensor to Python."
                )
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
