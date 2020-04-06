from typing import Optional

import storch
import torch
from torch.distributions import Distribution, Gumbel
import itertools


class SampleWithoutReplacementMethod(storch.Method):
    def __init__(self, plate_name: str, k: int):
        super().__init__(plate_name)
        self.k = k
        self.log_probs = None
        self.perturbed_log_probs = None

    def reset(self):
        self.log_probs = None
        self.perturbed_log_probs = None

    def _sample_tensor(
        self, distr: Distribution, parents: [storch.Tensor], plates: [storch.Plate]
    ) -> (torch.Tensor, int):
        # Perform stochastic beam search given the log-probs and perturbed log probs so far.
        # Samples k values from this distribution so that all total configurations are unique.
        # TODO: We have to keep track what samples are taken at each step for the backward pass.
        # Why? We need to think about what samples are discarded at some point because they are pruned away.
        # In the estimator they will still appear! So we'll have to think about that. They don't deserve a gradient
        # as they are only partial configurations and thus we don't know their loss.
        samples, self.log_probs, self.perturbed_log_probs = stochastic_beam_search(
            distr,
            self.k,
            len(plates),
            self.plate_name,
            self.log_probs,
            self.perturbed_log_probs,
        )
        return samples


def log1mexp(a: torch.Tensor) -> torch.Tensor:
    """See appendix A of http://jmlr.org/papers/v21/19-985.html.
    Numerically stable implementation of log(1-exp(a))"""
    r = torch.zeros_like(a)
    c = -0.693
    r[a > c] = (-a[a > c].expm1()).log()
    r[a <= c] = (-a[a <= c].exp()).log1p()
    return r


def log1pexp(a: torch.Tensor) -> torch.Tensor:
    """See appendix A of http://jmlr.org/papers/v21/19-985.html
    Numerically stable implementation of log(1+exp(a))
    TODO: The paper says there is a different case for a > 18... but the definition is invalid. need to check."""

    return (a.exp()).log1p()
    # r = torch.zeros_like(a)
    # c = 18
    # r[a >= c] = (-a[a>= c].exp()).log1p()
    # return r


def stochastic_beam_search(
    distribution: Distribution,
    k: int,
    amt_plates: int,
    # TODO: What is the plate dim? Should be the plate dim in the support tensor.
    plate_dim: str,
    log_probs: Optional[torch.Tensor],
    perturbed_log_probs: Optional[torch.Tensor],
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Implements Ancestral Gumbel-Top-k sampling with m=k, known as Stochastic Beam Search: http://jmlr.org/papers/v21/19-985.html.
    Sample from the distribution k items sequentially over the independent dimensions without replacement.
    Phi: Pass from method. Keeps track of current value of the perturbed log-probabilities. At storch.reset(), re-initialize.
    :param distribution:
    :param k:
    :param plate_dim Set to None if no dimension yet
    :return:
    """

    # TODO: This code is mostly taken from Expect(). May need to refactor to join these operations
    if not distribution.has_enumerate_support:
        raise ValueError(
            "Can only perform stochastic beam search for distributions with enumerable support."
        )

    support: torch.Tensor = distribution.enumerate_support(expand=True)

    sizes = support.shape[
        amt_plates + 1 : len(support.shape) - len(distribution.event_shape)
    ]

    ranges = []
    for size in sizes:
        ranges.append(range(size))

    support_non_expanded: torch.Tensor = distribution.enumerate_support(expand=False)
    yv_log_probs = distribution.log_prob(support_non_expanded)

    # Sample independent tensors in sequence
    # TODO: Check the shapes thoroughly here. There is the k dimension, which is the k best from the previous bfs layer
    # Then there is the current bfs layer, which is of shape |D_yv|. 
    samples = torch.zeros_like(support)
    if not plate_dim:
        samples.unsqueeze(0)
        plate_dim = 0
    for indices in itertools.product(ranges):
        if not log_probs:
            log_probs = yv_log_probs
            # First condition on max being 0:
            perturbed_log_probs = 0.0
        else:
            # Dunno how to shape this right, but this should be k times |support|=|D_yv|
            log_probs = log_probs + yv_log_probs
        # Sample |D_yv| Gumbel variables
        gumbel_d = Gumbel(loc=log_probs, scale=1.0)
        G_yv = gumbel_d.rsample()

        # Condition the Gumbel samples on the maximum of previous samples
        # TODO: again the shaping. Should take the maximum over |D_yv|
        Z = G_yv.max(-1)
        T = perturbed_log_probs
        vi = T - G_yv + log1mexp(G_yv - Z)
        cond_G_yv = T - vi.relu() - log1pexp(-vi.abs())

        # Select the k best
        # TODO: Argtop should be over both the k and the |D_yv| dimension, but is currently only over the last
        perturbed_log_probs, arg_top = torch.topk(cond_G_yv, k, dim=-1)
        log_probs = log_probs[arg_top]
        # Definitely wrong this shaping here, but that's the idea
        samples[..., *indices] = support_non_expanded[arg_top]
    return samples, log_probs, perturbed_log_probs
