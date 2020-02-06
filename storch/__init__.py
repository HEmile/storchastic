from .wrappers import deterministic, stochastic, cost
from .tensor import Tensor, DeterministicTensor, StochasticTensor
from .method import Method
from .inference import sample, backward, add_cost
from .util import print_graph
from .storch import cat
import storch.nn
_debug = False

