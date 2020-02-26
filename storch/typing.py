import torch
from storch.tensor import Tensor, StochasticTensor, DeterministicTensor
from storch.method.baseline import Baseline
from typing import Union, List, Tuple, Callable
from torch import Size
from torch.distributions import Bernoulli, Categorical, OneHotCategorical
# from storch.seq import BlackboxTensor

_size = Union[Size, List[int], Tuple[int, ...]]

'''
An AnyTensor object can be both a torch.Tensor or a storch.Tensor. Useful for code that allows both as input. 
'''
AnyTensor = Union[torch.Tensor, Tensor]
Dims = Union[int, _size]

# AnyBlackboxTensor = Union[BlackboxTensor, torch.Tensor]

DiscreteDistribution = Union[Bernoulli, Categorical, OneHotCategorical]

BaselineFactory = Callable[[StochasticTensor, DeterministicTensor], Baseline]