from .wrappers import (
    deterministic,
    make_left_broadcastable,
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
from .inference import backward, add_cost, reset, denote_independent, gather_samples, costs
from .util import print_graph

from .unique import unique, undo_unique
from .exceptions import IllegalStorchExposeError
from packaging import version

import storch.nn


import torch as _torch
import pkgutil
import sys

_debug = False
CHECK_DIFFERENTIABLE_PATH = False
# Hard-coded monkey patches: These do not support __torch_function__ (PyTorch version 1.5.0)
_torch.is_tensor = storch.is_tensor
_torch.Tensor.to = deterministic(_torch.Tensor.to)
_torch.Tensor.le = deterministic(_torch.Tensor.le)
_torch.Tensor.gt = deterministic(_torch.Tensor.gt)
# Cat currently does not support __torch_function__ https://github.com/pytorch/pytorch/issues/34294.
# However, monkey patching it makes storchastic incompatible with torchvision. Use storch.cat or storch.gather_samples instead
# _torch.cat = deterministic(_torch.cat)

# broadcast_all is not compatible with Tensor-likes... But a lot of Distributions code depends on it.
# Monkey patch every occurence of broadcast_all in the PyTorch code.
# This should not be necessesary from PyTorch 1.8.0. See https://github.com/pytorch/pytorch/pull/48169
if version.parse(torch.__version__) < version.parse("1.8.0"):
    _torch.distributions.utils.broadcast_all = deterministic(
        _torch.distributions.utils.broadcast_all
    )

    # Distributions import broadcast_all by name, meaning they refer to the non-monkey patched version.
    # Loop over every distribution to apply the monkey patch.
    for module_info in pkgutil.iter_modules(_torch.distributions.__path__):
        module = sys.modules.get("torch.distributions." + module_info.name)
        if "broadcast_all" in module.__dict__:
            module.broadcast_all = _torch.distributions.utils.broadcast_all

# Validating arguments is not compatible with Storchastic as it does instanceof torch.Tensor checks
# Necessary for PyTorch versions above 1.8.0
_torch.distributions.Distribution.set_default_validate_args(False)