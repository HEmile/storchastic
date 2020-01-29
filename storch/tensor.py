from __future__ import annotations
import torch
import storch
from typing import List

class Tensor:

    def __init__(self, tensor: torch.Tensor, is_stochastic: bool, sampling_method: storch.Method = None,
                 is_cost: bool = False):
        """
        Creates a storch.Tensor.
        :param tensor:
        :param is_stochastic:
        :param sampling_method:
        :param is_cost:
        """
        self._tensor = tensor
        self.stochastic = is_stochastic
        self.is_cost = is_cost
        self.sampling_method = sampling_method
        self._parents = []
        self._children = []

    @classmethod
    def deterministic(cls, tensor: torch.Tensor, is_cost: bool = False) -> Tensor:
        return cls(tensor, False, is_cost=is_cost)

    @classmethod
    def stochastic(cls, tensor: torch.Tensor, sampling_method: storch.Method) -> Tensor:
        return cls(tensor, True, sampling_method=sampling_method)

    def _add_parents(self, parents: List[Tensor]) -> None:
        for p in parents:
            self._parents.append(p)
            p._children.append(self)

    def __str__(self):
        t = "Stochastic" if self.stochastic else \
            ("Cost" if self.is_cost else "Deterministic")
        return t + " " + str(self._tensor)
