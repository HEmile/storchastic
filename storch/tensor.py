from __future__ import annotations
import torch
import storch
from torch.distributions import Distribution
from collections import deque


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
            differentiable_link = has_backwards_path(self._tensor, p._tensor)
            self._parents.append((p, differentiable_link))
            p._children.append((self, differentiable_link))
        self._children = []
        self.event_shape = tensor.shape[len(batch_links):]
        self.batch_links = batch_links

    def __str__(self):
        t = "Stochastic" if self.stochastic else ("Cost" if self.is_cost else "Deterministic")
        return t + " " + str(self._tensor.shape)

    def _walk(self, expand_fn, depth_first=True, only_differentiable=False, repeat_visited=False, walk_fn=lambda x: x):
        visited = set()
        if depth_first:
            S = [self]
            while S:
                v = S.pop()
                if repeat_visited or v not in visited:
                    yield walk_fn(v)
                    visited.add(v)
                    for w, d in expand_fn(v):
                        if d or not only_differentiable:
                            S.append(w)
        else:
            queue = deque()
            visited.add(self)
            queue.append(self)
            while queue:
                v = queue.popleft()
                yield walk_fn(v)
                for w, d in expand_fn(v):
                    if (repeat_visited or w not in visited) and (d or not only_differentiable):
                        visited.add(w)
                        queue.append(w)

    def walk_parents(self, depth_first=True, only_differentiable=False, repeat_visited=False, walk_fn=lambda x:x):
        return self._walk(lambda p: p._parents, depth_first, only_differentiable, repeat_visited, walk_fn)

    def walk_children(self, depth_first=True, only_differentiable=False, repeat_visited=False, walk_fn=lambda x: x):
        return self._walk(lambda p: p._children, depth_first, only_differentiable, repeat_visited, walk_fn)

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def is_cost(self) -> bool:
        return False

    @property
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size(map(lambda s: s.n, self.batch_links))

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape


class DeterministicTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, parents, batch_links: [StochasticTensor], is_cost: bool):
        super().__init__(tensor, parents, batch_links)
        self._is_cost = is_cost
        if is_cost:
            storch.inference._cost_tensors.append(self)

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def is_cost(self) -> bool:
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
        self.grads = []
        self._accum_grads = {}
        self.distribution = distribution

    @property
    def stochastic(self):
        return True

    @property
    def requires_grad(self):
        return self._requires_grad

    def mean_grad(self):
        return torch.mean(self.grads[0])

from storch.util import has_backwards_path