from storch.sampling.seq import AncestralPlate
from typing import Optional

import torch

import storch

from storch.sampling.swor import SampleWithoutReplacement


class MAPOSample(SampleWithoutReplacement):
    def __init__(self, plate_name: str, buffer_size: int, eos=None):
        super().__init__(plate_name, buffer_size + 1, eos)
        self.buffer_size = buffer_size
        self.buffer = None  # Initialized as empty buffer, run swor to fill it

        def select_samples(
                self, perturbed_log_probs: storch.Tensor, joint_log_probs: storch.Tensor,
        ) -> (storch.Tensor, storch.Tensor):
            """
            Given the perturbed log probabilities and the joint log probabilities of the new options, select which one to
            use for the sample.
            :param perturbed_log_probs: plates x (k? * |D_yv|). Perturbed log-probabilities. k is present if first_sample.
            :param joint_log_probs: plates x k? x (k? * |D_yv|). Joint log probabilities of the options. k is present if first_sample.
            :param first_sample:
            :return:
            """

            if self.buffer is not None:
                # We can sample at most the amount of what we previous sampled, combined with every option in the current domain
                # That is: prev_amt_samples * |D_yv|.
                amt_samples = min(self.k, perturbed_log_probs.shape[-1])
                # Take the top k over conditional perturbed log probs
                # plates x amt_samples
                return torch.topk(perturbed_log_probs, amt_samples, dim=-1)
                pass
            else:
                return super().select_samples(perturbed_log_probs, joint_log_probs)


class RaoBlackwellizedSample(SampleWithoutReplacement):
    """
    Implements the rao-blackwellized estimator from https://arxiv.org/abs/1810.04777
    Reuses the Stochastic Beam Search for efficient computation as suggested in https://www.jmlr.org/papers/v21/19-985.html
    """
    def __init__(self, plate_name: str, k: int, eos=None):
        """

        Args:
            plate_name:
            k: Total amount of samples. k-1 will be summed over (highest probability samples), then 1 reinforce sample
        """
        super().__init__(plate_name, k, eos)

    def select_samples(
            self, perturbed_log_probs: storch.Tensor, joint_log_probs: storch.Tensor,
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        """
        Given the perturbed log probabilities and the joint log probabilities of the new options, select which one to
        use for the sample.
        :param perturbed_log_probs: plates x (k? * |D_yv|). Perturbed log-probabilities. k is present if first_sample.
        :param joint_log_probs: plates x (k? * |D_yv|). Joint log probabilities of the options. k is present if first_sample.
        :param first_sample:
        :return: perturbed log probs of chosen samples, joint log probs of chosen samples, index of chosen samples
        """

        amt_samples = min(self.k, perturbed_log_probs.shape[-1])
        joint_log_probs_sum_over, sum_over_index = torch.topk(joint_log_probs, min(self.k - 1, amt_samples), dim=-1)
        perturbed_l_p_sum_over = perturbed_log_probs.gather(dim=-1, index=sum_over_index)

        if amt_samples == self.k:
            # Set the perturbed log probs of the samples chosen through the beam search to a low value
            filtered_perturbed_log_probs = torch.scatter(perturbed_log_probs, -1, sum_over_index, -1e10)
            perturbed_l_p_sample, sample_index = torch.max(filtered_perturbed_log_probs, dim=-1)
            perturbed_l_p_sample, sample_index = perturbed_l_p_sample.unsqueeze(-1), sample_index.unsqueeze(-1)
            joint_log_probs_sample = joint_log_probs.gather(dim=-1, index=sample_index)
            return (torch.cat((perturbed_l_p_sum_over, perturbed_l_p_sample), dim=-1),
                     torch.cat((joint_log_probs_sum_over, joint_log_probs_sample), dim=-1),
                     torch.cat((sum_over_index, sample_index), dim=-1))

        return perturbed_l_p_sum_over, joint_log_probs_sum_over, sum_over_index


    def weighting_function(
        self, tensor: storch.StochasticTensor, plate: AncestralPlate
    ) -> Optional[storch.Tensor]:
        """
        Returns the weighting of the sample
        :param tensor:
        :param plate:
        :return:
        """
        amt_samples = plate.log_probs.shape[-1]
        if amt_samples == self.k:
            def _priv(log_probs: torch.Tensor):
                joint_probs = log_probs[..., :-1].exp()
                iw_sample = (1. - joint_probs.sum(dim=-1).detach()).unsqueeze(-1)
                weighting = torch.cat((joint_probs, iw_sample), dim=-1)
                return weighting
            return storch.deterministic(_priv)(plate.log_probs)

        return plate.log_probs.exp()
