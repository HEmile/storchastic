from .wrappers import (
    deterministic,
    stochastic,
    reduce,
    _exception_wrapper,
    _unpack_wrapper,
    ignore_wrapping,
)
from .tensor import Tensor, CostTensor, StochasticTensor, Plate, is_tensor

import storch.sampling
from .storch import *
import storch.method
import storch.typing
from .inference import backward, add_cost, reset, denote_independent, gather_samples
from .util import print_graph

from .unique import unique, undo_unique
from .exceptions import IllegalStorchExposeError

import storch.nn


import torch as _torch

_debug = False
# Hard-coded monkey patches: These do not support __torch_function__ (PyTorch version 1.5.0)
_torch.is_tensor = storch.is_tensor
_torch.Tensor.to = deterministic(_torch.Tensor.to)
_torch.Tensor.le = deterministic(_torch.Tensor.le)
_torch.Tensor.gt = deterministic(_torch.Tensor.gt)
# Cat currently does not support __torch_function__ https://github.com/pytorch/pytorch/issues/34294.
# However, monkey patching it makes storchastic incompatible with torchvision. Use storch.cat or storch.gather_samples instead
# _torch.cat = deterministic(_torch.cat)
