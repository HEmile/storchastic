from typing import Dict, Optional, List, Tuple, Union
from collections import deque

from pyro.distributions import (
    RelaxedOneHotCategoricalStraightThrough,
    RelaxedBernoulliStraightThrough,
)

from storch.tensor import Tensor, CostTensor, StochasticTensor, Plate, is_tensor
from torch.distributions import (
    Distribution,
    RelaxedOneHotCategorical,
    Categorical,
    OneHotCategorical,
    Bernoulli,
    RelaxedBernoulli,
    Gumbel,
)
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
            if is_tensor(p) and (not filter_requires_grad or p.requires_grad):
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
    visited = set()
    to_visit = deque()
    to_visit.append(grad)
    pop_func = to_visit.pop if depth_first else to_visit.popleft
    while to_visit:
        n = pop_func()
        yield n
        visited.add(n)
        for t, _ in n.next_functions:
            if t and t not in visited:
                to_visit.append(t)


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
        for param in get_distr_parameters(output.distribution).values():
            if has_backwards_path(param, input, depth_first):
                return True
        return False
    input = input._tensor
    if not input.requires_grad:
        return False
    if isinstance(output, Tensor):
        output = output._tensor
    if output is input:
        # This can happen if the input is a parameter of the output distribution
        return True
    if not output.grad_fn:
        return False
    for p in walk_backward_graph(output, depth_first):
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


def rsample_gumbel(distr: Distribution, n: int,) -> torch.Tensor:
    gumbel_distr = Gumbel(distr.logits, 1)
    return gumbel_distr.rsample((n,))


def rsample_gumbel_softmax(
    distr: Distribution,
    n: int,
    temperature: torch.Tensor,
    straight_through: bool = False,
) -> torch.Tensor:
    if isinstance(distr, (Categorical, OneHotCategorical)):
        if straight_through:
            gumbel_distr = RelaxedOneHotCategoricalStraightThrough(
                temperature, probs=distr.probs
            )
        else:
            gumbel_distr = RelaxedOneHotCategorical(temperature, probs=distr.probs)
    elif isinstance(distr, Bernoulli):
        if straight_through:
            gumbel_distr = RelaxedBernoulliStraightThrough(
                temperature, probs=distr.probs
            )
        else:
            gumbel_distr = RelaxedBernoulli(temperature, probs=distr.probs)
    else:
        raise ValueError("Using Gumbel Softmax with non-discrete distribution")
    return gumbel_distr.rsample((n,))


def split(
    tensor: Tensor,
    plate: Plate,
    *,
    amt_slices: Optional[int] = None,
    slices: Optional[List[slice]] = None,
    create_plates=True
) -> Tuple[Tensor, ...]:
    """
    Splits the plate dimension on the tensor into several tensors and returns those tensors. Note: It removes the
    tensors from the computation graph and therefore should only be used when creating estimators, when logging or debugging, or
    if you know what you're doing.
    """
    if not slices:
        slice_length = int(plate.n / amt_slices)
        slices = []
        for i in range(amt_slices):
            slices.append(slice(i * slice_length, (i + 1) * slice_length))

    new_plates = tensor.plates.copy()

    index = tensor.get_plate_dim_index(plate.name)
    plates_index = new_plates.index(plate)
    if not create_plates and tensor.plate_dims != index + 1:
        for _plate in reversed(tensor.plates):
            if _plate.n > 1:
                # Overwrite old plate
                new_plates.remove(_plate)
                new_plates[plates_index] = _plate
                break
    empty_indices = [None] * (index - 1)
    sliced_tensors = []
    for _slice in slices:
        indices = empty_indices + [_slice]
        new_tensor = tensor._tensor[indices]
        if create_plates:
            n = _slice.stop - _slice.start
            final_plates = new_plates.copy()
            final_plates[plates_index] = Plate(plate.name, n, plate.parents)
            if n == 1:
                new_tensor = new_tensor.squeeze(index)
            new_tensor = Tensor(new_tensor, [], final_plates, tensor.name)
        else:
            new_tensor = Tensor(
                new_tensor.transpose(index, tensor.plate_dims - 1),
                [],
                new_plates,
                tensor.name,
            )
        sliced_tensors.append(new_tensor)
    return tuple(sliced_tensors)
