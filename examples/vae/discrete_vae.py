from typing import Tuple, List

from examples.vae import VAE, main
import torch.nn as nn
import torch
import storch
from torch.distributions import OneHotCategorical


class DiscreteVAE(VAE):
    def initialize_param_layers(
        self, latents: int, prev_layer: int
    ) -> (nn.Module, nn.Module):
        fc3 = nn.Linear(prev_layer, latents * 10)
        fc4 = nn.Linear(latents * 10, prev_layer)
        return fc3, fc4

    def initialize_method(self, args) -> storch.method.Method:
        if args.method == "gumbel":
            return storch.method.GumbelSoftmax()
        elif args.method == "gumbel_straight":
            return storch.method.GumbelSoftmax(straight_through=True)
        elif args.method == "score":
            return storch.method.ScoreFunction(baseline_factory=args.baseline)
        elif args.method == "expect":
            return storch.method.Expect()

    def prior(self, shape: List[int]):
        return OneHotCategorical(probs=torch.ones(shape + [10]) / (1.0 / 10.0))

    def variational_posterior(self, logits):
        return OneHotCategorical(logits=logits)

    def logits_to_params(self, logits, latents):
        return logits.reshape(logits.shape[:-1] + (latents, 10))

    def shape_latent(self, z, latents):
        return z.reshape(z.shape[:-2] + (latents * 10,))


if __name__ == "__main__":
    main(DiscreteVAE)