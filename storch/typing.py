from __future__ import annotations
import torch
import storch
from typing import Union, List, Tuple, Callable
from torch import Size
from torch.distributions import Bernoulli, Categorical, OneHotCategorical

# from storch.seq import BlackboxTensor

AnyTensor = Union[storch.Tensor, torch.Tensor]

_size = Union[Size, List[int], Tuple[int, ...]]

Dims = Union[int, _size]

# AnyBlackboxTensor = Union[BlackboxTensor, torch.Tensor]

DiscreteDistribution = Union[Bernoulli, Categorical, OneHotCategorical]


_index = Union[str, int, storch.Plate]
_indices = Union[List[_index], _index]

_plates = Union[List[Union[storch.Plate, str]], storch.Plate, str]
