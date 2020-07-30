from examples.vae import VAE, main
import torch.nn as nn
import torch
import storch
from torch.distributions import Distribution, Bernoulli
from storch.method import (
    ScoreFunctionWOR,
    UnorderedSetEstimator,
    RELAX,
    REBAR,
    ScoreFunction,
    GumbelSoftmax,
    Expect,
)
from torch.distributions.utils import clamp_probs


class BernoulliVAE(VAE):
    def initialize_param_layers(
        self, latents: int, prev_layer: int
    ) -> (nn.Module, nn.Module):
        fc3 = nn.Linear(prev_layer, latents)
        fc4 = nn.Linear(latents, prev_layer)
        return fc3, fc4

    def initialize_method(self, args) -> storch.method.Method:
        if args.method == "gumbel":
            return GumbelSoftmax("z", n_samples=args.samples)
        if args.method == "gumbel_sparse":
            return storch.method.GumbelSparseMax("z", n_samples=args.samples)
        if args.method == "gumbel_entmax":
            return storch.method.GumbelEntmax(
                "z", n_samples=args.samples, adaptive=True
            )
        elif args.method == "gumbel_straight":
            return GumbelSoftmax("z", n_samples=args.samples, straight_through=True)
        elif args.method == "gumbel_wor":
            return storch.method.UnorderedSetGumbelSoftmax("z", k=args.samples)
        elif args.method == "score":
            return ScoreFunction(
                "z", n_samples=args.samples, baseline_factory=args.baseline
            )
        elif args.method == "expect":
            return Expect("z")
        elif args.method == "relax":
            return RELAX("z", n_samples=args.samples, in_dim=(args.latents,))
        elif args.method == "rebar":
            return REBAR("z", n_samples=args.samples)
        elif args.method == "relax_rebar":
            return RELAX(
                "z", n_samples=args.samples, in_dim=(args.latents,), rebar=True
            )
        elif args.method == "score_wor":
            return ScoreFunctionWOR("z", k=args.samples)
        elif args.method == "unordered":
            return UnorderedSetEstimator("z", k=args.samples)
        elif args.method == "arm":
            return storch.method.ARM("z", n_samples=args.samples)
        elif args.method == "disarm":
            return storch.method.DisARM("z", n_samples=args.samples)
        else:
            raise ValueError("Invalid method passed to program arguments.")

    def prior(self, posterior: Distribution):
        return Bernoulli(probs=torch.ones_like(posterior.probs) / 2.0)

    def variational_posterior(self, logits: torch.Tensor):
        return Bernoulli(probs=clamp_probs(logits.sigmoid()))

    def logits_to_params(self, logits, latents):
        return logits

    def shape_latent(self, z, latents):
        if z.dtype == torch.bool:
            return z.float()
        return z

    def name(self):
        return "bernoulli vae"


if __name__ == "__main__":
    main(BernoulliVAE)
