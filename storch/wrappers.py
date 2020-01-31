import storch
import torch
from storch.tensor import Tensor, DeterministicTensor

_context_stochastic = False
_context_deterministic = False
_stochastic_parents = None

def _unwrap(*args, **kwargs):
    unwrapped = []
    p = []
    for a in args:
        if isinstance(a, Tensor):
            unwrapped.append(a._tensor)
            p.append(a)
        else:
            if not isinstance(a, torch.Tensor):
                print("Unwrapping of values other than tensors is currently not supported", a)
            unwrapped.append(a)

    if len(kwargs.values()) > 0:
        print("Unwrapping of kwargs is currently not supported")
    return tuple(unwrapped), p


def _process_deterministic(o, parents, is_cost):
    if isinstance(o, Tensor):
        raise RuntimeError("Creation of storch Tensor within deterministic context")
    if isinstance(o, torch.Tensor):
        t = DeterministicTensor(o, is_cost)
        t._add_parents(parents)
        return t
    raise NotImplementedError("Handling of other types of return values is currently not implemented")


def _deterministic(fn, is_cost):
    def wrapper(*args, **kwargs):
        if storch.wrappers._context_deterministic:
            if is_cost:
                raise RuntimeError("Cannot call storch.cost from within a deterministic context.")

            # We are already in a deterministic context, no need to wrap or unwrap as only the outer dependencies matter
            return fn(*args, **kwargs)
        args, parents = _unwrap(*args, **kwargs)

        if not parents:
            return fn(*args, **kwargs)
        storch.wrappers._context_deterministic = True
        outputs = fn(*args, **kwargs)
        if type(outputs) is tuple:
            outputs = []
            for o in outputs:
               outputs.append(_process_deterministic(o, parents, is_cost))
        else:
            outputs = _process_deterministic(outputs, parents, is_cost)
        storch.wrappers._context_deterministic = False
        return outputs
    return wrapper


def cost(fn):
    return _deterministic(fn, True)


def deterministic(fn):
    return _deterministic(fn, False)


def _process_stochastic(output):
    if isinstance(output, Tensor):
        if not output.stochastic:
            # The Tensor was created by calling @deterministic within a stochastic context.
            # This means that we have to conservatively assume it is dependent on the parents
            output._add_parents(storch.wrappers._stochastic_parents)
        return output
    if isinstance(output, torch.Tensor):
        t = DeterministicTensor(output, False)
        t._add_parents(storch.wrappers._stochastic_parents)
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
        # Save the parents
        args, parents = _unwrap(*args, **kwargs)
        storch.wrappers._stochastic_parents = parents

        outputs = fn(*args, **kwargs)

        # Add parents to the outputs
        if type(outputs) is tuple:
            processed_outputs = []
            for o in outputs:
                processed_outputs.append(_process_stochastic(o))
        else:
            processed_outputs = _process_stochastic(outputs)
        storch.wrappers._context_stochastic = False
        storch.wrappers._stochastic_parents = None
        return processed_outputs
    return wrapper


