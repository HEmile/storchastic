from __future__ import annotations

from typing import Union, Any, Tuple, List, Optional, Dict, Callable

import storch
import torch
from collections.abc import Iterable, Mapping
from functools import wraps
from storch.exceptions import IllegalStorchExposeError
from contextlib import contextmanager

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
        and not storch.is_tensor(a)
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
    elif isinstance(a, Mapping):
        max_event_dim = 0
        for _a in a.values():
            max_event_dim = max(
                max_event_dim, _collect_parents_and_plates(_a, parents, plates)
            )
        return max_event_dim
    elif is_iterable(a):
        max_event_dim = 0
        for _a in a:
            max_event_dim = max(
                max_event_dim, _collect_parents_and_plates(_a, parents, plates)
            )
        return max_event_dim
    return 0


def _unsqueeze_and_unwrap(
    a: Any,
    multi_dim_plates: [storch.Plate],
    align_tensors: bool,
    l_broadcast: bool,
    expand_plates: bool,
    flatten_plates: bool,
    event_dims: int,
):
    if isinstance(a, storch.Tensor):
        if not align_tensors:
            return a._tensor

        for plate in multi_dim_plates:
            # Used in storch.method.sampling.AncestralPlate
            a = plate.on_unwrap_tensor(a)

        tensor = a._tensor
        # Automatically **RIGHT** broadcast. Ensure each tensor has an equal amount of event dims by inserting dimensions to the right
        # TODO: What do we think about this design?
        # TODO: The storch.tensor._getitem_level == 0 check prevents right-broadcasting for __getitem__ and __setitem__... Seems hacky
        if l_broadcast and a.event_dims < event_dims:
            tensor = tensor[(...,) + (None,) * (event_dims - a.event_dims)]
        # It can be possible that the ordering of the plates does not align with the ordering of the inputs.
        # This part corrects this.
        amt_recognized = 0
        links: [storch.Plate] = a.multi_dim_plates()

        for i, plate in enumerate(multi_dim_plates):
            if plate in links:
                if plate != links[amt_recognized]:
                    # The plate is also in the tensor, but not in the ordering expected. So switch that ordering
                    j = links.index(plate)
                    tensor = tensor.transpose(j, amt_recognized)
                    links[amt_recognized], links[j] = links[j], links[amt_recognized]
                amt_recognized += 1

        # Add singleton dimensions on missing plates
        plate_dims = []
        for i, plate in enumerate(multi_dim_plates):
            if plate not in a.plates:
                tensor = tensor.unsqueeze(i)
                plate_dims.append(plate.n)
            else:
                # Make sure to use a's plate size here. It's actually possible they are different in ancestral plates!
                plate_dims.append(tensor.shape[i])

        # Optionally expand the singleton dimensions to the plate size
        if expand_plates:
            tensor = tensor.expand(tuple(plate_dims) + tensor.shape[len(plate_dims) :])
        # Optionally flatten the plate dimensions to a single batch dimension
        if flatten_plates:
            assert expand_plates
            tensor = tensor.reshape((-1,) + tensor.shape[len(plate_dims) :])

        return tensor
    elif isinstance(a, Mapping):
        d = {}
        for k, _a in a.items():
            d[k] = _unsqueeze_and_unwrap(
                _a,
                multi_dim_plates,
                align_tensors,
                l_broadcast,
                expand_plates,
                flatten_plates,
                event_dims,
            )
        return d
    elif is_iterable(a):
        l = []
        for _a in a:
            l.append(
                _unsqueeze_and_unwrap(
                    _a,
                    multi_dim_plates,
                    align_tensors,
                    l_broadcast,
                    expand_plates,
                    flatten_plates,
                    event_dims,
                )
            )
        if isinstance(a, tuple):
            return tuple(l)
        return l
    else:
        return a


def _prepare_args(
    fn_args,
    fn_kwargs,
    unwrap=True,
    align_tensors=True,
    l_broadcast=True,
    expand_plates=False,
    flatten_plates=False,
    dim: Optional[str] = None,
    dims: Optional[Union[str, List[str]]] = None,
) -> (List, Dict, [storch.Tensor], [storch.Plate]):
    """
    Prepares the input arguments of the wrapped function:
    - Unwrap the input arguments from storch.Tensors to normal torch.Tensors so they can be used in any torch function
    - Align plate dimensions for automatic broadcasting
    - Add (singleton) plate dimensions for plates that are not present
    - Right-broadcast event dimensions for automatic broadcasting
    - Superclasses of Plate specific input handling

    :param fn_args: List of non-keyword arguments to the wrapped function
    :param fn_kwargs: Dictionary of keyword arguments to the wrapped function
    :param unwrap: Whether to unwrap the arguments to their torch.Tensor counterpart (default: True)
    :param align_tensors: Whether to automatically align the input arguments (default: True)
    :param l_broadcast: Whether to automatically left-broadcast (default: True)
    :param expand_plates: Instead of adding singleton dimensions on non-existent plates, this will
    add the plate size itself (default: False) flatten_plates sets this to True automatically.
    :param flatten_plates: Flattens the plate dimensions into a single batch dimension if set to true.
    This can be useful for functions that are written to only work for tensors with a single batch dimension.
    Note that outputs are unflattened automatically. (default: False)
    :param dim: Replaces the dim input in fn_kwargs by the plate dimension corresponding to the given string (optional)
    :param dims: Replaces the dims input in fn_kwargs by the plate dimensions corresponding to the given strings (optional)
    :param self_wrapper: storch.Tensor that wraps a
    :return: Handled non-keyword arguments, handled keyword arguments, list of parents, list of plates
    """
    parents: [storch.Tensor] = []
    plates: [storch.Plate] = []
    max_event_dim = max(
        # Collect parent tensors and plates
        _collect_parents_and_plates(fn_args, parents, plates),
        _collect_parents_and_plates(fn_kwargs, parents, plates),
    )

    # Allow plates to filter themselves from being collected.
    plates = list(filter(lambda p: p.on_collecting_args(plates), plates))

    # Get the list of plates with size larger than 1 for the unsqueezing of tensors
    multi_dim_plates = []
    for plate in plates:
        if plate.n > 1:
            multi_dim_plates.append(plate)

    if dim:
        for i, plate in enumerate(multi_dim_plates):
            if dim == plate.name:
                i_dim = i
                break
        fn_kwargs["dim"] = i_dim
    if dims:
        dimz = []
        for dim in dims:
            for i, plate in enumerate(multi_dim_plates):
                if dim == plate.name:
                    dimz.append(i_dim)
                    break
            raise ValueError("Missing plate dimension" + dim)
        fn_kwargs["dims"] = dimz

    if unwrap:
        expand_plates = expand_plates or flatten_plates
        # Unsqueeze and align batched dimensions so that batching works easily.
        unsqueezed_args = []
        for t in fn_args:
            unsqueezed_args.append(
                _unsqueeze_and_unwrap(
                    t,
                    multi_dim_plates,
                    align_tensors,
                    l_broadcast,
                    expand_plates,
                    flatten_plates,
                    max_event_dim,
                )
            )
        unsqueezed_kwargs = {}
        for k, v in fn_kwargs.items():
            unsqueezed_kwargs[k] = _unsqueeze_and_unwrap(
                v,
                multi_dim_plates,
                align_tensors,
                l_broadcast,
                expand_plates,
                flatten_plates,
                max_event_dim,
            )
        return unsqueezed_args, unsqueezed_kwargs, parents, plates
    return fn_args, fn_kwargs, parents, plates


def _prepare_outputs_det(
    o: Any,
    parents: [storch.Tensor],
    plates: [storch.Plate],
    name: str,
    index: int,
    unflatten_plates,
):
    if o is None:
        return None, index
    if isinstance(o, storch.Tensor):
        if o.stochastic:
            raise RuntimeError(
                "Creation of stochastic storch Tensor within deterministic context"
            )
        # TODO: Does this require shape checking? Parent/Plate checking?
        return o, index + 1
    if isinstance(o, torch.Tensor):  # Explicitly _not_ a storch.Tensor
        if unflatten_plates:
            plate_dims = tuple([plate.n for plate in plates if plate.n > 1])
            o = o.reshape(plate_dims + o.shape[1:])
        t = storch.Tensor(o, parents, plates, name=name + str(index))
        return t, index + 1
    if is_iterable(o):
        outputs = []
        for _o in o:
            t, index = _prepare_outputs_det(
                _o, parents, plates, name, index, unflatten_plates=unflatten_plates
            )
            outputs.append(t)
        if isinstance(o, tuple):
            return tuple(outputs), index
        return outputs, index
    raise NotImplementedError(
        "Handling of other types of return values is currently not implemented: ", o
    )


def _handle_deterministic(
    fn,
    fn_args,
    fn_kwargs,
    reduce_plates: Optional[Union[str, List[str]]] = None,
    flatten_plates: bool = False,
    **wrapper_kwargs
):
    if storch.wrappers._context_stochastic:
        raise NotImplementedError(
            "It is currently not allowed to open a deterministic context in a stochastic context"
        )
        # TODO check if we can re-add this
        # if storch.wrappers._context_deterministic > 0:
        #     if is_cost:
        #         raise RuntimeError("Cannot call storch.cost from within a deterministic context.")

        # TODO: This is currently uncommented and it will in fact unwrap. This was required because it was, eg,
        # possible to open a deterministic context, passing distributions with storch.Tensors as parameters,
        # then doing computations on these parameters. This is because these storch.Tensors will not be unwrapped
        # in the deterministic context as the unwrapping only considers lists.
        # # We are already in a deterministic context, no need to wrap or unwrap as only the outer dependencies matter
        # return fn(*args, **kwargs)

    new_fn_args, new_fn_kwargs, parents, plates = _prepare_args(
        fn_args, fn_kwargs, flatten_plates=flatten_plates, **wrapper_kwargs
    )
    if not parents:
        return fn(*fn_args, **fn_kwargs)
    args = new_fn_args
    kwargs = new_fn_kwargs

    storch.wrappers._context_deterministic += 1

    try:
        outputs = fn(*args, **kwargs)
    finally:
        storch.wrappers._context_deterministic -= 1

    if storch.wrappers._ignore_wrap:
        return outputs
    if reduce_plates:
        if isinstance(reduce_plates, str):
            reduce_plates = [reduce_plates]
        plates = [p for p in plates if p.name not in reduce_plates]

    outputs = _prepare_outputs_det(
        outputs, parents, plates, fn.__name__, 1, unflatten_plates=flatten_plates
    )[0]
    return outputs


def _deterministic(
    fn, reduce_plates: Optional[Union[str, List[str]]] = None, **wrapper_kwargs
):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal reduce_plates
        return _handle_deterministic(fn, args, kwargs, reduce_plates, **wrapper_kwargs)

    return wrapper


def deterministic(fn: Optional[Callable] = None, **kwargs):
    """
    Wraps the input function around a deterministic storch wrapper.
    This wrapper unwraps :class:`~storch.Tensor` objects to :class:`~torch.Tensor` objects, aligning the tensors
    according to the plates, then runs `fn` on the unwrapped Tensors.

    Args:
        fn: Optional function to wrap. If None, this returns another wrapper that accepts a function that will be instantiated
        by the given kwargs.
        unwrap: Set to False to prevent unwrapping :class:`~storch.Tensor` objects.
    Returns:
        Callable: The wrapped function `fn`.
    """
    if fn:
        return _deterministic(fn, **kwargs)
    return lambda _f: _deterministic(_f, **kwargs)


def reduce(fn, plates: Union[str, List[str]]):
    """
    Wraps the input function around a deterministic storch wrapper.
    This wrapper unwraps :class:`~storch.Tensor` objects to :class:`~torch.Tensor` objects, aligning the tensors
    according to the plates, then runs `fn` on the unwrapped Tensors. It will reduce the plates given by `plates`.

    Args:
        fn (Callable): Function to wrap.
    Returns:
        Callable: The wrapped function `fn`.
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
        args, kwargs, parents, plates = _prepare_args(*args, **kwargs)

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


def _unpack_wrapper(fn, self: Optional[storch.Tensor] = None):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if self:
            args = list(args)
            args.insert(0, self)
        new_args = []
        for a in args:
            if isinstance(a, storch.Tensor):
                new_args.append(a._tensor)
            else:
                new_args.append(a)
        return fn(*tuple(new_args), **kwargs)

    return wrapper


@contextmanager
def ignore_wrapping():
    storch.wrappers._ignore_wrap = True
    yield
    storch.wrappers._ignore_wrap = False
