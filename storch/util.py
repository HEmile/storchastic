from storch.tensor import Tensor
from torch.distributions import Distribution
import torch

def print_graph(node: Tensor, depth_first = False):
    counters = [1, 1, 1]
    names = {}

    def get_name(node, names):
        if node in names:
            return names[node]
        if node.stochastic:
            name = "s" + str(counters[0])
            counters[0] += 1
        elif node.is_cost:
            name = "c" + str(counters[1])
            counters[1] += 1
        else:
            name = "d" + str(counters[2])
            counters[2] += 1
        names[node] = name
        return name

    def pretty_print(node):
        name = get_name(node, names)
        print(name, node)
        for p, differentiable in node._parents:
            edge = "-D->" if differentiable else "-X->"
            print(get_name(p, names) + edge + name)

    pretty_print(node)
    for p, _ in node.walk_parents(depth_first):
        pretty_print(p)


def get_distr_parameters(d: Distribution, filter_requires_grad=True) -> [torch.Tensor]:
    params = []
    for k in d.arg_constraints:
        try:
            p = getattr(d, k)
            if isinstance(p, torch.Tensor) and (not filter_requires_grad or p.requires_grad):
                params.append(p)
        except AttributeError:
            pass
    return params


def _walk_backward_graph(grad, depth_first=True):
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


def has_differentiable_path(output: torch.Tensor, input: torch.Tensor, depth_first=True):
    for p in walk_backward_graph(output, depth_first):
        if hasattr(p, "variable") and p.variable is input:
            return True
        elif p is input.grad_fn:
            return True
    return False