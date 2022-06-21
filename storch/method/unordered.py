from typing import Optional, Tuple

import storch
from storch.method.multi_sample_reinforce import SeqMethod
from storch.sampling import UnorderedSet, GumbelSoftmaxWOR
from storch.util import magic_box


class UnorderedSetEstimator(SeqMethod):
    """
    Implements the Unordered Set REINFORCE Estimator with baseline from https://openreview.net/forum?id=rklEj2EFvB
    (Wouter Kool et al, ICLR 2020)
    It uses Stochastic Beam Search to sample from sequences of discrete random variables without replacement.
    Then it normalizes these samples, treating them as unordered samples without replacement, that is, as sets.
    The baseline takes a weighted average over the cost using these samples.

    This implementation uses the trapezoid rule to compute the leave-one-out ratio for the plate weighting.
    This numerical integration can be demanding. One can choose to decrease num_int_points to trade off computation
    time for precision.
    """

    def __init__(
        self,
        plate_name: str,
        k: int,
        use_baseline: bool = True,
        exact_integration: bool = False,
        num_int_points: int = 1000,
        a: float = 5.0,
        eos=None,
    ):
        """
        Creates an Unordered Set Estimator method.
        :param plate_name: The name of the ancestral plate for the sequence to be sampled.
        :param k: The amount of samples to take using stochastic beam search.
        :param use_baseline: Whether to use the built-in baseline (see Equation 17 and 18 of https://openreview.net/forum?id=rklEj2EFvB)
        :param exact_integration: Whether to use exact integration instead of numerical integration. Currently not implemented
        :param num_int_points: How many points to use in the trapezoid rule for the numerical integration in the
        computation of the leave-one-out ratio.
        :param a: Hyperparameter for numerical stable computation of the leave-one-out ratio. The default is recommended
        in https://openreview.net/forum?id=rklEj2EFvB, but could be tuned if NaNs pop up.
        """
        super().__init__(
            plate_name,
            UnorderedSet(
                plate_name,
                k,
                use_baseline,
                exact_integration,
                num_int_points,
                a,
                eos=eos,
            ),
        )
        self.use_baseline = use_baseline

    def estimator(
        self, tensor: storch.StochasticTensor, cost_node: storch.CostTensor
    ) -> Tuple[
        Optional[storch.Tensor], Optional[storch.Tensor]
    ]:
        # Note: We automatically multiply with leave-one-out ratio in the plate reduction
        plate = None
        for _p in cost_node.plates:
            if _p.name == self.plate_name:
                plate = _p
                break
        if not self.use_baseline:
            return plate.log_probs * cost_node.detach()

        # Subtract the 'average' cost of other samples, keeping in mind that samples are not independent.
        # plates x k
        baseline = storch.sum(
            (plate.log_probs + plate.log_snd_leave_one_out).exp() * cost_node, plate
        )
        # Make sure the k dimension is recognized as batch dimension
        baseline = storch.Tensor(
            baseline._tensor, [baseline], baseline.plates + [plate]
        )
        return plate.log_probs, (1 - magic_box(plate.log_probs)) * baseline.detach()


class UnorderedSetGumbelSoftmax(SeqMethod):
    def __init__(
        self,
        plate_name: str,
        k: int,
        initial_temperature=1.0,
        min_temperature=1.0e-4,
        annealing_rate=1.0e-5,
        eos=None,
    ):
        super().__init__(
            plate_name,
            GumbelSoftmaxWOR(
                plate_name, k, initial_temperature, min_temperature, annealing_rate, eos
            ),
        )
