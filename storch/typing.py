from __future__ import annotations
import torch
import storch
from storch.method.baseline import Baseline
from typing import Union, List, Tuple, Callable
from torch import Size
from torch.distributions import Bernoulli, Categorical, OneHotCategorical

# from storch.seq import BlackboxTensor

AnyTensor = Union[storch.Tensor, torch.Tensor]

_size = Union[Size, List[int], Tuple[int, ...]]

Dims = Union[int, _size]

# AnyBlackboxTensor = Union[BlackboxTensor, torch.Tensor]

DiscreteDistribution = Union[Bernoulli, Categorical, OneHotCategorical]

BaselineFactory = Callable[[storch.StochasticTensor, storch.CostTensor], Baseline]
