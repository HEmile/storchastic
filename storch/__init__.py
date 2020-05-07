_debug = False

from .wrappers import (
    deterministic,
    stochastic,
    reduce,
    _exception_wrapper,
    _unpack_wrapper,
    ignore_wrapping,
)
from .tensor import Tensor, CostTensor, StochasticTensor, Plate, is_tensor
from .method import *
from .inference import backward, add_cost, reset, denote_independent, gather_samples
from .util import print_graph
from .storch import *
import storch.typing
import storch.nn

import torch

torch.is_tensor = storch.is_tensor
torch.Tensor.to = deterministic(torch.Tensor.to)
