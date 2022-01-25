from __future__ import annotations
from typing import Optional, Callable, List

import torch
from torch.distributions import Distribution
from abc import ABC, abstractmethod
import storch
from storch import Plate


class SamplingMethod(ABC, torch.nn.Module):
    def __init__(self, plate_name: str):
        super().__init__()
        self.reset()
        self.plate_name = plate_name

    def reset(self) -> None:
        pass

    def forward(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        requires_grad: bool,
    ) -> (storch.StochasticTensor, Plate):
        return self.sample(distr, parents, plates, requires_grad)

    @abstractmethod
    def sample(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        requires_grad: bool,
    ) -> (storch.StochasticTensor, Plate):
        pass

    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        """
        Weight by the size of the sample. Overload this if your sampling method uses some kind of weighting
        of the different events, like importance sampling or computing the expectation.
        If None is returned, it is assumed the samples are iid monte carlo samples.

        This method is called from storch.method.Method.sample, and it is not needed to manually call this on created plates
        """
        return self.mc_weighting_function(tensor, plate)

    def mc_weighting_function(
        self, tensor: storch.StochasticTensor, plate: Plate
    ) -> Optional[storch.Tensor]:
        return None

    def update_parameters(
        self,
        result_triples: [(storch.StochasticTensor, storch.CostTensor, torch.Tensor)],
    ):
        pass

    def mc_sample(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        amt_samples: int,
    ) -> torch.Tensor:
        # TODO: Why does this ignore amt_samples?
        return distr.sample((amt_samples,))

    def set_mc_sample(
        self,
        new_sample_func: Callable[
            [Distribution, [storch.Tensor], [Plate], int], torch.Tensor
        ],
    ) -> SamplingMethod:
        """
        Override storch.Method specific sampling functions.
        This is called when initializing a storch.Method that has slightly different MC sampling semantics
        (for example, reparameterization instead of normal sampling).
        This allows for compatibility of different `storch.Method`'s with different `storch.sampling.Method`'s.
        """
        self.mc_sample = new_sample_func
        return self

    def set_mc_weighting_function(
        self,
        new_weighting_func: Callable[
            [storch.StochasticTensor, Plate], Optional[storch.Tensor]
        ],
    ) -> SamplingMethod:
        """
        Override storch.Method specific weighting functions.
        This is called when initializing a storch.Method that has slightly different MC weighting semantics
        (for example, REBAR that weights some samples differently).
        This allows for compatibility of different `storch.Method`'s with different `storch.sampling.Method`'s.
        """
        self.mc_weighting_function = new_weighting_func
        return self

    def on_plate_already_present(self, plate: Plate):
        raise ValueError(
            "Cannot create stochastic tensor with name "
            + plate.name
            + ". A parent sample has already used this name. Use a different name for this sample."
        )


class MonteCarlo(SamplingMethod):
    """
    Monte Carlo sampling methods use simple sampling methods that take n independent samples.
    Unlike complex ancestral sampling methods such as SampleWithoutReplacementMethod, the sampling behaviour is not dependent
    on earlier samples in the stochastic computation graph (but the distributions are!).
    """

    def __init__(self, plate_name: str, n_samples: int = 1):
        super().__init__(plate_name)
        self.n_samples = n_samples

    def sample(
        self,
        distr: Distribution,
        parents: [storch.Tensor],
        plates: [Plate],
        requires_grad: bool,
    ) -> (storch.StochasticTensor, Plate):
        plate = None
        for _plate in plates:
            if _plate.name == self.plate_name:
                plate = _plate
                break
        n_samples = 1 if plate else self.n_samples
        with storch.ignore_wrapping():
            tensor = self.mc_sample(distr, parents, plates, n_samples)
        plate_size = tensor.shape[0]
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)

        if not plate:
            plate = Plate(self.plate_name, plate_size, plates.copy())
            plates.insert(0, plate)

        if isinstance(tensor, storch.Tensor):
            tensor = tensor._tensor

        s_tensor = storch.StochasticTensor(
            tensor, parents, plates, self.plate_name, plate_size, distr, requires_grad,
        )
        return s_tensor, plate
