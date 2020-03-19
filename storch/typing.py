from __future__ import annotations
import torch
from storch.tensor import Tensor, StochasticTensor, CostTensor
from storch.method.baseline import Baseline
from typing import Union, List, Tuple, Callable
from torch import Size
from torch.distributions import Bernoulli, Categorical, OneHotCategorical

# from storch.seq import BlackboxTensor

_size = Union[Size, List[int], Tuple[int, ...]]

Dims = Union[int, _size]

# AnyBlackboxTensor = Union[BlackboxTensor, torch.Tensor]

DiscreteDistribution = Union[Bernoulli, Categorical, OneHotCategorical]

BaselineFactory = Callable[[StochasticTensor, CostTensor], Baseline]
