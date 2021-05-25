from examples.vae import VAE, main
import torch.nn as nn
import torch
import storch
from torch.distributions import OneHotCategorical, Distribution
from storch.method import (
    ScoreFunctionWOR,
    UnorderedSetEstimator,
    RELAX,
    REBAR,
    ScoreFunction,
    GumbelSoftmax,
    Expect,
)


class DiscreteVAE(VAE):
    def initialize_param_layers(
        self, latents: int, prev_layer: int
    ) -> (nn.Module, nn.Module):
        fc3 = nn.Linear(prev_layer, latents * 10)
        fc4 = nn.Linear(latents * 10, prev_layer)
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
        # TODO: Seems to be broken!
        elif args.method == "gumbel_wor":
            return storch.method.UnorderedSetGumbelSoftmax("z", k=args.samples)
        elif args.method == "score":
            return ScoreFunction(
                "z", n_samples=args.samples, baseline_factory=args.baseline
            )
        elif args.method == "expect":
            return Expect("z")
        elif args.method == "relax":
            return RELAX("z", n_samples=args.samples, in_dim=(args.latents, 10))
        elif args.method == "rebar":
            return REBAR("z", n_samples=args.samples)
        elif args.method == "relax_rebar":
            return RELAX(
                "z", n_samples=args.samples, in_dim=(args.latents, 10), rebar=True
            )
        elif args.method == "score_wor":
            return ScoreFunctionWOR("z", k=args.samples)
        elif args.method == "unordered":
            return UnorderedSetEstimator("z", k=args.samples)
        else:
            raise ValueError("Invalid method passed to program arguments.")

    def prior(self, posterior: Distribution):
        return OneHotCategorical(probs=torch.ones_like(posterior.probs) / 10.0)

    def variational_posterior(self, logits: torch.Tensor):
        return OneHotCategorical(probs=logits.softmax(dim=-1))
        # return OneHotCategorical(logits=logits)

    def logits_to_params(self, logits: torch.Tensor, latents: int) -> torch.Tensor:
        return logits.reshape(logits.shape[:-1] + (latents, 10))

    def shape_latent(self, z: torch.Tensor, latents: int) -> torch.Tensor:
        return z.reshape(z.shape[:-2] + (latents * 10,))

    def name(self) -> str:
        return "discrete_vae"


if __name__ == "__main__":
    main(DiscreteVAE)
