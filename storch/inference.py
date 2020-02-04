from torch.distributions import Distribution
from storch.tensor import Tensor, StochasticTensor
from storch.method import Method, Infer, ScoreFunction
from storch.wrappers import _stochastic_parents
from storch.util import topological_sort, print_graph, get_distr_parameters
import storch
_cost_tensors = []


def sample(distr: Distribution, method: Method = None, n: int = 1) -> Tensor:
    if not method:
        if distr.has_rsample:
            method = Infer()
        else:
            method = ScoreFunction()
    params = get_distr_parameters(distr, filter_requires_grad=True)
    tensor = method.sample(distr, n)
    if n == 1:
        tensor = tensor.squeeze(0)
    plates = storch.wrappers._plate_links.copy()
    s_tensor = StochasticTensor(tensor, storch.wrappers._stochastic_parents, method, plates, distr, len(params) > 0, n)
    return s_tensor


def backward(retain_graph=False, debug=False):
    costs = storch.inference._cost_tensors
    if debug:
        print_graph(costs)
    Q = {}

    for c in costs:
        for p in c.walk_parents():
            if p not in Q:
                Q[p] = []
            Q[p].append(c)

    print(Q)

    nodes = topological_sort(costs)

    # for v in nodes:
    #     for w, d in v._parents:
    #         if not w.stochastic:

    if not retain_graph:
        storch.inference._cost_tensors = []