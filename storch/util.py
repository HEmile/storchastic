import storch
from torch.distributions import Distribution
import torch
from storch.tensor import Tensor


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
        for p in node._parents:
            print(get_name(p, names) + "->" + name)

    pretty_print(node)
    for p in node.walk_parents(depth_first):
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
