import storch
import torch

_CONTEXT_STOCHASTIC = False
_CONTEXT_DETERMINISTIC = False


def _unwrap(*args, **kwargs):
    unwrapped = []
    p = []
    for a in args:
        if isinstance(a, storch.Tensor):
            unwrapped.append(a._tensor)
            p.append(a)
        else:
            if not isinstance(a, torch.Tensor):
                print("Unwrapping of values other than tensors is currently not supported")
            unwrapped.append(a)

    if len(kwargs.values()) > 0:
        print("Unwrapping of kwargs is currently not supported")
    return tuple(unwrapped), p


def _process_deterministic(o, parents, is_cost):
    if isinstance(o, storch.Tensor):
        if o not in parents:
            raise RuntimeError("Creation of storch Tensor within deterministic context")
        o.is_cost = o.is_cost or is_cost
        return o
    if isinstance(o, torch.Tensor):
        t = storch.Tensor.deterministic(o, is_cost)
        t._add_parents(parents)
        return t
    raise NotImplementedError("Handling of other types of return values is currently not implemented")


def _deterministic(fn, is_cost):
    def wrapper(*args, **kwargs):
        if storch.wrappers._CONTEXT_STOCHASTIC:
            raise RuntimeError("Cannot create a deterministic context from within a stochastic context.")
        if storch.wrappers._CONTEXT_DETERMINISTIC:
            if is_cost:
                raise RuntimeError("Cannot call storch.cost from within a deterministic context.")

            # We are already in a deterministic context, no need to wrap or unwrap as only the outer dependencies matter
            return fn(*args, **kwargs)
        storch.wrappers._CONTEXT_DETERMINISTIC = True
        args, parents = _unwrap(*args, **kwargs)
        outputs = fn(*args, **kwargs)
        if type(outputs) is tuple:
            outputs = []
            for o in outputs:
               outputs.append(_process_deterministic(o, parents, is_cost))
        else:
            outputs = _process_deterministic(outputs, parents, is_cost)
        storch.wrappers._CONTEXT_DETERMINISTIC = False
        return outputs
    return wrapper


def cost(fn):
    return _deterministic(fn, True)


def deterministic(fn):
    return _deterministic(fn, False)


def _process_stochastic(output, parents) -> None:
    if not isinstance(output, storch.Tensor) or not output.stochastic:
        raise TypeError("All outputs of functions wrapped in @storch.stochastic "
                        "should be stochastic storch Tensors. At " + str(output))
    output._add_parents(parents)


def stochastic(fn):
    """
    Applies `fn` to the `inputs`. `fn` should return one or multiple `storch.Tensor`s.
    `fn` should not call `storch.stochastic` or `storch.deterministic`. `inputs` can include `storch.Tensor`s.
    :param fn:
    :return:
    """
    def wrapper(*args, **kwargs):
        if storch.wrappers._CONTEXT_STOCHASTIC or storch.wrappers._CONTEXT_DETERMINISTIC:
            raise RuntimeError("Cannot call storch.stochastic from within a stochastic or deterministic context.")
        storch.wrappers._CONTEXT_STOCHASTIC = True
        args, parents = _unwrap(*args, **kwargs)
        outputs = fn(*args, **kwargs)
        if type(outputs) is tuple:
            for o in outputs:
                _process_stochastic(o, parents)
        else:
            _process_stochastic(outputs, parents)
        storch.wrappers._CONTEXT_STOCHASTIC = False
        return outputs
    return wrapper


