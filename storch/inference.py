from torch.distributions import Distribution
from storch.tensor import Tensor
from storch.method import Method, Infer, ScoreFunction
from storch.wrappers import _stochastic_parents
from storch.util import topological_sort, print_graph
import storch
_cost_tensors = []


def sample(distr: Distribution, method: Method = None, n: int = 1) -> Tensor:
    if not method:
        if distr.has_rsample:
            method = Infer()
        else:
            method = ScoreFunction()
    s_tensor = method.sample(distr, n)
    if _stochastic_parents:
        s_tensor._add_parents(_stochastic_parents)
    return s_tensor


def backward(retain_graph=False, debug=False):
    if debug:
        print_graph(storch.inference._cost_tensors)

    if not retain_graph:
        storch.inference._cost_tensors = []