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
            differentiable_link = has_differentiable_path(self._tensor, p._tensor)
            self._parents.append((p, differentiable_link))
            p._children.append((self, differentiable_link))

    def __str__(self):
        t = "Stochastic" if self.stochastic else "Deterministic"
        return t + " " + str(self._tensor)

    def _walk(self, collection, recur_fn, depth_first=False):
        if depth_first:
            for p in collection:
                for _p in recur_fn(p):
                    yield _p
                yield p
        else:
            for p in collection:
                yield p
            for p in collection:
                for _p in recur_fn(p):
                    yield _p

    def walk_parents(self, depth_first=False):
        return self._walk(self._parents, lambda p: p[0].walk_parents(depth_first), depth_first)

    def walk_children(self, depth_first=False):
        return self._walk(self._children, lambda p: p[0].walk_children(depth_first), depth_first)

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

from storch.util import has_differentiable_path