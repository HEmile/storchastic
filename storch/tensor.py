from __future__ import annotations
import torch
import storch
from typing import List
from torch.distributions import Distribution


class Tensor:

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
        self._parents = []
        self._children = []

    def _add_parents(self, parents: List[Tensor]) -> None:
        for p in parents:
            self._parents.append(p)
            p._children.append(self)

    def __str__(self):
        t = "Stochastic" if self.stochastic else "Deterministic"
        return t + " " + str(self._tensor)

    @property
    def stochastic(self):
        return False

    @property
    def is_cost(self):
        return False

    @property
    def requires_grad(self):
        return self._tensor.requires_grad


class DeterministicTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, is_cost: bool):
        super().__init__(tensor)
        self._is_cost = is_cost

    @property
    def stochastic(self):
        return False

    @property
    def is_cost(self):
        return self._is_cost


class StochasticTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, sampling_method: storch.Method, distribution: Distribution,
                 requires_grad: bool):
        super().__init__(tensor)
        self.sampling_method = sampling_method
        self._requires_grad = requires_grad
        self.grads = None
        self.distribution = distribution

    @property
    def stochastic(self):
        return True

    @property
    def requires_grad(self):
        return self._requires_grad

    def mean_grad(self):
        return torch.mean(self.grads)