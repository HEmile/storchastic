from __future__ import annotations
import torch
import storch
from torch.distributions import Distribution


class Tensor:

    def __init__(self, tensor: torch.Tensor, parents: [Tensor], batch_links: [StochasticTensor]):
        for i, plate in enumerate(batch_links):
            if len(tensor.shape) <= i or not tensor.shape[i] == plate.n:
                raise ValueError(
                    "Storch Tensors should take into account their surrounding plates. Violated at dimension " + str(i)
                    + " and plate size " + str(plate.n) + ". Instead, it was " + str(tensor.shape[i]))

        self._tensor = tensor
        self._parents = []
        for p in parents:
            if p.is_cost:
                raise ValueError("Cost nodes cannot have children.")
            differentiable_link = has_differentiable_path(self._tensor, p._tensor)
            self._parents.append((p, differentiable_link))
            p._children.append((self, differentiable_link))
        self._children = []
        self.event_shape = tensor.shape[len(batch_links):]
        self.batch_links = batch_links

    def __str__(self):
        t = "Stochastic" if self.stochastic else "Deterministic"
        return t + " " + str(self._tensor)

    def _walk(self, collection, recur_fn, depth_first=False, only_differentiable=False):
        if depth_first:
            for p, d in collection:
                for _p, d in recur_fn(p):
                    if d or not only_differentiable:
                        yield _p, d
                if d or not only_differentiable:
                    yield p, d
        else:
            for p, d in collection:
                if d or not only_differentiable:
                    yield p, d
            for p in collection:
                for _p, d in recur_fn(p):
                    if d or not only_differentiable:
                        yield _p, d

    def walk_parents(self, depth_first=False, only_differentiable=False):
        return self._walk(self._parents, lambda p: p[0].walk_parents(depth_first), depth_first, only_differentiable)

    def walk_children(self, depth_first=False, only_differentiable=False):
        return self._walk(self._children, lambda p: p[0].walk_children(depth_first), depth_first, only_differentiable)

    @property
    def stochastic(self):
        return False

    @property
    def is_cost(self):
        return False

    @property
    def requires_grad(self):
        return self._tensor.requires_grad

    @property
    def batch_shape(self):
        return torch.Size(map(lambda s: s.n, self.batch_links))


class DeterministicTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, parents, batch_links: [StochasticTensor], is_cost: bool):
        super().__init__(tensor, parents, batch_links)
        self._is_cost = is_cost
        if is_cost:
            storch.inference._cost_tensors.append(self)

    @property
    def stochastic(self):
        return False

    @property
    def is_cost(self):
        return self._is_cost


class StochasticTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, parents, sampling_method: storch.Method, batch_links: [StochasticTensor],
                 distribution: Distribution, requires_grad: bool, n: int):
        if n > 1:
            batch_links.insert(0, self)
        self.n = n
        super().__init__(tensor, parents, batch_links)
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