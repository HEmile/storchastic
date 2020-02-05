from storch.tensor import Tensor, DeterministicTensor
from torch.distributions import Distribution
import torch

def print_graph(costs: [DeterministicTensor]):
    nodes = topological_sort(costs)
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

    for node in nodes:
        name = get_name(node, names)
        print(name, node)
        for p, differentiable in node._parents:
            edge = "-D->" if differentiable else "-X->"
            print(get_name(p, names) + edge + name)


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


def has_backwards_path(output: torch.Tensor, input: torch.Tensor, depth_first=False):
    """

    :param output:
    :param input:
    :param depth_first: Initialized to False as we are usually doing this only for small distances between tensors.
    :return:
    """
    if not output.grad_fn:
        return False
    for p in walk_backward_graph(output, depth_first):
        if hasattr(p, "variable") and p.variable is input:
            return True
        elif p is input.grad_fn:
            return True
    return False


def has_differentiable_path(output:Tensor, input: Tensor):
    for c in input.walk_children(only_differentiable=True):
        if c is output:
            return True
    return False


def topological_sort(costs: [DeterministicTensor]) -> [Tensor]:
    """
    Implements reverse kahn's algorithm
    :param costs:
    :return:
    """
    for c in costs:
        if not c.is_cost or c._children:
            raise ValueError("The inputs of the topological sort should only contain cost nodes.")
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
            children.remove(n)
            if not children:
                s.append(p)
    return l
