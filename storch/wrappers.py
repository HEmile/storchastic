from __future__ import annotations
import storch
import torch
from collections.abc import Iterable, Mapping

# from storch.tensor import Tensor, DeterministicTensor, StochasticTensor

_context_stochastic = False
_context_deterministic = False
_stochastic_parents = []
_context_name = None
_plate_links = []


def is_iterable(a):
    return isinstance(a, Iterable) and not isinstance(a, torch.Tensor) and not isinstance(a, str)


def _collect_parents_and_plates(a, parents: [storch.Tensor], plates: [storch.StochasticTensor]):
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


def _unsqueeze_and_unwrap(a, plates: [storch.StochasticTensor]):
    if isinstance(a, storch.Tensor):
        tensor = a._tensor

        # It can be possible that the ordering of the plates does not align with the ordering of the inputs.
        # This part corrects this.
        amt_recognized = 0
        links = a.batch_links.copy()
        for i, plate in enumerate(plates):
            if plate in a.batch_links:
                if plate is not links[amt_recognized]:
                    # The plate is also in the tensor, but not in the ordering expected. So switch that ordering
                    j = links.index(plate)
                    tensor = tensor.transpose(j, amt_recognized)
                    links[amt_recognized], links[j] = links[j], links[amt_recognized]
                amt_recognized += 1

        for i, plate in enumerate(plates):
            if plate not in a.batch_links:
                tensor = tensor.unsqueeze(i)
        return tensor
    elif is_iterable(a):
        l = []
        for _a in a:
            l.append(_unsqueeze_and_unwrap(_a, plates))
        if isinstance(a, tuple):
            return tuple(l)
        return l
    elif isinstance(a, Mapping):
        d = {}
        for k, _a in a.items():
            d[k] = _unsqueeze_and_unwrap(_a, plates)
        return d
    else:
        return a


def _unwrap(*args, **kwargs):
    parents = []
    plates: [storch.StochasticTensor] = []

    # Collect parent tensors and plates
    _collect_parents_and_plates(args, parents, plates)
    _collect_parents_and_plates(kwargs, parents, plates)

    storch.wrappers._plate_links = plates

    # Unsqueeze and align batched dimensions so that batching works easily.
    unsqueezed_args = []
    for t in args:
        unsqueezed_args.append(_unsqueeze_and_unwrap(t, plates))
    unsqueezed_kwargs = {}
    for k, v in kwargs.items():
        unsqueezed_kwargs[k] = _unsqueeze_and_unwrap(v, plates)
    return unsqueezed_args, unsqueezed_kwargs, parents, plates


def _process_deterministic(o, parents, plates, is_cost, name):
    if o is None:
        return
    if isinstance(o, storch.Tensor):
        raise RuntimeError("Creation of storch Tensor within deterministic context")
    if isinstance(o, torch.Tensor):
        t = storch.DeterministicTensor(o, parents, plates, is_cost, name=name)
        if is_cost and t.event_shape != ():
            # TODO: Make sure the o.size() check takes into account the size of the sample.
            raise ValueError(
                "Event shapes (ie, non batched dimensions) of cost nodes have to be single floating point numbers. ")
        return t
    raise NotImplementedError("Handling of other types of return values is currently not implemented: ", o)


def _deterministic(fn, is_cost):
    def wrapper(*args, **kwargs):
        if storch.wrappers._context_stochastic:
            # TODO check if we can re-add this
            raise NotImplementedError(
                "It is currently not allowed to open a deterministic context in a stochastic context")
        if storch.wrappers._context_deterministic:
            if is_cost:
                raise RuntimeError("Cannot call storch.cost from within a deterministic context.")

            # We are already in a deterministic context, no need to wrap or unwrap as only the outer dependencies matter
            return fn(*args, **kwargs)
        args, kwargs, parents, plates = _unwrap(*args, **kwargs)

        if not parents and not is_cost:
            return fn(*args, **kwargs)
        storch.wrappers._context_deterministic = True
        outputs = fn(*args, **kwargs)
        if is_iterable(outputs):
            n_outputs = []
            for o in outputs:
                n_outputs.append(_process_deterministic(o, parents, plates, is_cost, fn.__name__))
            outputs = n_outputs
        else:
            outputs = _process_deterministic(outputs, parents, plates, is_cost, fn.__name__)
        storch.wrappers._context_deterministic = False
        return outputs

    return wrapper


def deterministic(fn):
    return _deterministic(fn, False)


def cost(fn):
    return _deterministic(fn, True)


def _process_stochastic(output, parents, plates):
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
        if storch.wrappers._context_stochastic or storch.wrappers._context_deterministic:
            raise RuntimeError("Cannot call storch.stochastic from within a stochastic or deterministic context.")
        storch.wrappers._context_stochastic = True
        storch.wrappers._context_name = fn.__name__
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
        return processed_outputs

    return wrapper
