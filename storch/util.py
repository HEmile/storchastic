from typing import Dict

from storch.tensor import Tensor, CostTensor, StochasticTensor
from torch.distributions import Distribution
import torch


def print_graph(costs: [CostTensor]):
    nodes = topological_sort(costs)
    counters = {"s": 1, "c": 1, "d": 1}
    names = {}

    def get_name(node, names):
        if node in names:
            return names[node]
        if node.name:
            if node.name not in counters:
                counters[node.name] = 1
                name = node.name + "[" + str(0) + "]"
            else:
                name = node.name + "[" + str(counters[node.name]) + "]"
                counters[node.name] += 1
        elif node.stochastic:
            name = "s" + "[" + str(counters["s"]) + "]"
            counters["s"] += 1
        elif node.is_cost:
            name = "c" + "[" + str(counters["c"]) + "]"
            counters["c"] += 1
        else:
            name = "d" + "[" + str(counters["d"]) + "]"
            counters["d"] += 1
        names[node] = name
        return name

    for node in nodes:
        name = get_name(node, names)
        print(name, node)
    for node in nodes:
        name = get_name(node, names)
        for p, differentiable in node._parents:
            edge = "-D->" if differentiable else "-X->"
            print(get_name(p, names) + edge + name)


def get_distr_parameters(
    d: Distribution, filter_requires_grad=True
) -> Dict[str, torch.Tensor]:
    params = {}
    for k in d.arg_constraints:
        try:
            p = getattr(d, k)
            if isinstance(p, torch.Tensor) and (
                not filter_requires_grad or p.requires_grad
            ):
                params[k] = p
        except AttributeError:
            from storch import _debug

            if _debug:
                print(
                    "Attribute",
                    k,
                    "was not added because we could not get the attribute from the object.",
                )
            pass
    return params


def _walk_backward_graph(grad, depth_first=True):
    if not grad:
        return
    if depth_first:
        for t, _ in grad.next_functions:
            if not t:
                continue
            yield t
            for o in _walk_backward_graph(t, depth_first):
                yield o
    else:
        for t, _ in grad.next_functions:
            yield t
        for t, _ in grad.next_functions:
            for o in _walk_backward_graph(t, depth_first):
                yield o


def walk_backward_graph(tensor: torch.Tensor, depth_first=True):
    if not tensor.grad_fn:
        raise ValueError("Can only walk backward over graphs with a gradient function.")
    return _walk_backward_graph(tensor.grad_fn, depth_first)


def has_backwards_path(output: Tensor, input: Tensor, depth_first=False):
    """
    Returns true if the gradient functions of the torch.Tensor underlying output is connected to the input tensor.
    This is only run once to compute the possibility of links between two storch.Tensor's. The result is saved into the
    parent links on storch.Tensor's.
    :param output:
    :param input:
    :param depth_first: Initialized to False as we are usually doing this only for small distances between tensors.
    :return:
    """

    if isinstance(output, StochasticTensor):
        outputs = get_distr_parameters(output.distribution)
    else:
        outputs = {None: output._tensor}
    input = input._tensor
    if not input.requires_grad:
        return False
    for o in outputs.values():
        if isinstance(o, Tensor):
            o = o._tensor
        if o is input:
            # This can happen if the input is a parameter of the output distribution
            return True
        if not o.grad_fn:
            continue
        for p in walk_backward_graph(o, depth_first):
            if hasattr(p, "variable") and p.variable is input:
                return True
            elif input.grad_fn and p is input.grad_fn:
                return True
    return False


def has_differentiable_path(output: Tensor, input: Tensor):
    for c in input.walk_children(only_differentiable=True):
        if c is output:
            return True
    return False


def topological_sort(costs: [CostTensor]) -> [Tensor]:
    """
    Implements reverse kahn's algorithm
    :param costs:
    :return:
    """
    for c in costs:
        if not c.is_cost or c._children:
            raise ValueError(
                "The inputs of the topological sort should only contain cost nodes."
            )
    l = []
    s = costs.copy()
    edges = {}
    while s:
        n = s.pop()
        l.append(n)
        for (p, _) in n._parents:
            if p in edges:
                children = edges[p]
            else:
                children = list(map(lambda ch: ch[0], p._children))
                edges[p] = children
            c = -1
            for i, _c in enumerate(children):
                if _c is n:
                    c = i
                    break
            del children[c]
            if not children:
                s.append(p)
    return l


def tensor_stats(tensor: torch.Tensor):
    return "shape {} mean {:.3f} std {:.3f} max {:.3f} min {:.3f}".format(
        tuple(tensor.shape),
        tensor.mean().item(),
        tensor.std().item(),
        tensor.max().item(),
        tensor.min().item(),
    )


def reduce_mean(tensor: torch.Tensor, keep_dims: [int]):
    if len(keep_dims) == tensor.ndim:
        return tensor
    sum_out_dims = list(range(tensor.ndim))
    for dim in keep_dims:
        sum_out_dims.remove(dim)
    return tensor.mean(sum_out_dims)


def reduce_plate(tensor: Tensor, plate: [str, int, torch.Tensor], index: int):
    plate_weighting = plate[2]
    # Case: The weight is a single number. First sum, then multiply with the weight (usually taking the mean)
    if plate_weighting is None or plate_weighting.ndim == 0:
        return tensor.sum(index) * plate_weighting
    # Case: The weight is a vector of numbers equal to batch dimension. First multiply the
    else:
        onez = [1] * (tensor.ndim - 1)
        onez.insert(index, -1)
        onez = tuple(onez)
        plate_weighting = plate_weighting.view(*onez)
        return (tensor * plate_weighting).sum(index)
