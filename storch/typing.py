import torch
import storch
from typing import Union, List, Tuple
from torch import Size
from torch.distributions import Bernoulli, Categorical, OneHotCategorical
from storch.seq import BlackboxTensor

_size = Union[Size, List[int], Tuple[int, ...]]

'''
An AnyTensor object can be both a torch.Tensor or a storch.Tensor. Useful for code that allows both as input. 
'''
AnyTensor = Union[torch.Tensor, storch.Tensor]
Dims = Union[int, _size]

AnyBlackboxTensor = Union[BlackboxTensor, torch.Tensor]

DiscreteDistribution = Union[Bernoulli, Categorical, OneHotCategorical]