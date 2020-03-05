from typing import Callable

from .wrappers import deterministic, stochastic, reduce, _exception_wrapper, _unpack_wrapper
from .tensor import Tensor, CostTensor, StochasticTensor
from .method import *
from .inference import backward, add_cost, reset, denote_independent
from .util import print_graph
from .storch import *
import storch.typing
import storch.nn
_debug = False

from inspect import isclass
from .excluded_init import _excluded_init, _exception_init, _unwrap_only

# Wrap all (common) normal PyTorch functions. This is required because they call
# C code. The returned tensors from the C code can only automatically be wrapped
# by wrapping the functions that call the C code.
# Ignore the first 35 as they are not related to tensors
let_through = False
for m, v in torch.__dict__.items():
    if m == "_adaptive_avg_pool2d":
        let_through = True
    if not let_through or not isinstance(v, Callable):
        continue
    if m in _exception_init:
        torch.__dict__[m] = _exception_wrapper(v)
    elif m in _unwrap_only:
        torch.__dict__[m] = _unpack_wrapper(v)
    elif not isclass(v) and not str.startswith(m, "_cufft_") \
            and not str.startswith(m, "rand") and m not in _excluded_init:
        torch.__dict__[m] = deterministic(v)

for m, v in torch.nn.functional.__dict__.items():
    if isinstance(v, Callable):
        torch.nn.functional.__dict__[m] = deterministic(v)
