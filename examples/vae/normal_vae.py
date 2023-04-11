from typing import List

from examples.vae import VAE, main
import torch.nn as nn
import torch
import storch
from torch.distributions import Normal, Distribution
from pyro.ops.indexing import Vindex


class NormalVAE(VAE):
    def initialize_param_layers(
        self, latents: int, prev_layer: int
    ) -> (nn.Module, nn.Module):
        # *2 because we compute both the mean and stdev
        fc3 = nn.Linear(prev_layer, latents * 2)
        fc4 = nn.Linear(latents, prev_layer)
        return fc3, fc4

    def initialize_method(self, args) -> storch.method.Method:
        if args.method == "reparameterization":
            return storch.method.Reparameterization("z", n_samples=args.samples)
        elif args.method == "lax":
            return storch.method.LAX("z", n_samples=args.samples, in_dim=args.latents)
        elif args.method == "score":
            return storch.method.ScoreFunction(
                "z", n_samples=args.samples, baseline_factory=args.baseline
            )

    def prior(self, posterior: Normal) -> Distribution:
        return Normal(torch.zeros_like(posterior.loc), torch.ones_like(posterior.scale))

    def variational_posterior(self, params) -> Distribution:
        mean, std = params
        return Normal(mean, std)

    def logits_to_params(self, logits, latents):
        # mean, std
        return logits[..., :latents], torch.exp(0.5 * logits[..., latents:])

    def name(self):
        return "normal_vae"


if __name__ == "__main__":
    main(NormalVAE)
