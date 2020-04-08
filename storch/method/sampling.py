from typing import Optional

import storch
import torch
from torch.distributions import Distribution, Gumbel
import itertools
from storch.method.method import Method


class SampleWithoutReplacementMethod(Method):
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


def right_expand_as(tensor, expand_as):
    diff = expand_as.ndim - tensor.ndim
    return tensor[(...,) + (None,) * diff].expand(
        (-1,) * tensor.ndim + expand_as.shape[tensor.ndim :]
    )


def stochastic_beam_search(
    distribution: Distribution,
    k: int,
    amt_plates: int,
    plate_dim: str,
    joint_log_probs: Optional[torch.Tensor],
    perturbed_log_probs: Optional[torch.Tensor],
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Sample k events from the distribution without replacement.

    Implements Ancestral Gumbel-Top-k sampling with m=k, known as Stochastic Beam Search: http://jmlr.org/papers/v21/19-985.html.
    This sample from the distribution k items sequentially over the independent dimensions without replacement
    :param distribution: The distribution to sample from
    :param k: The amount of
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
        ranges.append(list(range(size)))

    support_non_expanded: torch.Tensor = distribution.enumerate_support(expand=False)
    d_log_probs = distribution.log_prob(support_non_expanded)

    # Sample independent tensors in sequence
    # TODO: Check the shapes thoroughly here. There is the k dimension, which is the k best from the previous bfs layer
    # Then there is the current bfs layer, which is of shape |D_yv|.
    # TODO: What if fewer than k is being sampled (ie |D_yv|<k)? That could 'underfil' the samples.
    # if not plate_dim:
    #     samples.unsqueeze(0)
    #     plate_dim = 0
    sampled_support_indices = support.new_zeros(
        size=(k,) + support.shape[1:-1], dtype=torch.long
    )
    # Iterate over the different (conditionally) independent samples being taken
    for indices in itertools.product(*ranges):
        # List of indices for the next sample
        indexing = (slice(None),) * (amt_plates + 1) + indices
        # Log probabilities of the different options for this sample step
        yv_log_probs = d_log_probs[indexing]
        if joint_log_probs is None:
            joint_log_probs = yv_log_probs
            # First condition on max being 0:
            perturbed_log_probs = 0.0
            first_sample = True
        else:
            # Returns |D_yv| x k x ...
            joint_log_probs = joint_log_probs.unsqueeze(0) + yv_log_probs.unsqueeze(1)
            first_sample = False

        # Sample |D_yv| Gumbel variables
        gumbel_d = Gumbel(loc=joint_log_probs, scale=1.0)
        G_yv = gumbel_d.rsample()

        # Condition the Gumbel samples on the maximum of previous samples
        # Index 0 is always |D_yv|
        Z = G_yv.max(0)[0]
        T = perturbed_log_probs
        vi = T - G_yv + log1mexp(G_yv - Z)
        cond_G_yv = T - vi.relu() - log1pexp(-vi.abs())

        # Select the k best, then use the result to select the log probabilities
        # TODO: Argtop should be over both the k and the |D_yv| dimension, but is currently only over the last. Should reshape it to be over both.
        if first_sample:
            perturbed_log_probs, arg_top = torch.topk(cond_G_yv, k, dim=0)
            joint_log_probs = joint_log_probs.gather(dim=0, index=arg_top)
            sampled_support_indices[indexing] = arg_top
        else:
            print(sampled_support_indices[:, 1])
            cond_G_yv = cond_G_yv.reshape((-1,) + cond_G_yv.shape[2:])
            # Gather corresponding joint log probabilities
            perturbed_log_probs, arg_top = torch.topk(cond_G_yv, k, dim=0)
            joint_log_probs = joint_log_probs.reshape(
                (-1,) + joint_log_probs.shape[2:]
            ).gather(dim=0, index=arg_top)
            # Keep track of what parents were sampled for the arg top
            # TODO: Also keep track of what other parents were sampled for the argtop
            chosen_parents = right_expand_as(
                arg_top.remainder(k), sampled_support_indices
            )
            sampled_support_indices = sampled_support_indices.gather(
                dim=0, index=chosen_parents
            )
            chosen_samples = arg_top / k
            sampled_support_indices[indexing] = chosen_samples

            print(sampled_support_indices[:, 1])
            print(perturbed_log_probs)

    expanded_indices = right_expand_as(sampled_support_indices, support)
    samples = support.gather(dim=0, index=expanded_indices)
    return samples, joint_log_probs, perturbed_log_probs
