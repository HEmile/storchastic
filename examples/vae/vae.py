"""
Edited from https://github.com/pytorch/examples/blob/master/vae/main.py
Reproduce experiments from Kool 2020 and Yin 2019
"""

from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from storch import deterministic
import storch
from torch.distributions import Distribution


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args

        self.latents = args.latents
        self.samples = args.samples
        self.sampling_method = self.initialize_method(args)
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)

        self.fc3, self.fc4 = self.initialize_param_layers(self.latents, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.activation = lambda x: F.leaky_relu(x, negative_slope=0.1)

    def initialize_param_layers(
        self, latents: int, prev_layer: int
    ) -> (nn.Module, nn.Module):
        pass

    def initialize_method(self, args) -> storch.method.Method:
        pass

    @deterministic
    def encode(self, x: storch.Tensor):
        """
        Define the encoding step. As it is a purely deterministic operation, we can tag it using @determinstic to slightly
        improve the performance.
        :param x: The input data
        :return: The encoded logits
        """
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        return self.fc3(h2)

    @deterministic
    def decode(self, z: storch.Tensor):
        """
        Define the decoding step. As it is a purely deterministic operation, we can tag it using @determinstic to slightly
        improve the performance.
        :param z: The sampled latent variables
        :return: Logits for the MNIST problem
        """
        z = self.shape_latent(z, self.latents)
        h3 = self.activation(self.fc4(z))
        h4 = self.activation(self.fc5(h3))
        return self.fc6(h4).sigmoid()

    def forward(
        self, x: storch.Tensor
    ) -> (storch.Tensor, storch.Tensor, storch.Tensor):
        logits = self.encode(x)
        params = self.logits_to_params(logits, self.latents)
        var_posterior = self.variational_posterior(params)
        prior = self.prior(var_posterior)

        KLD = self.KLD(var_posterior, prior)
        storch.add_cost(KLD, "KL-divergence")
        z = self.sampling_method(var_posterior)
        return self.decode(z), KLD, z

    def prior(self, posterior: Distribution) -> Distribution:
        """
        Returns the distribution that represents the prior distribution.
        :param posterior: The posterior distribution that is used.
        """
        pass

    def variational_posterior(self, params) -> Distribution:
        """
        Returns the variational distribution given its input parameters
        """
        pass

    def logits_to_params(self, logits: storch.Tensor, latents: int) -> storch.Tensor:
        """
        Converts the encoded logits into the parameters for the variational posterior
        """
        return logits

    def shape_latent(self, z: storch.Tensor, latents: int) -> storch.Tensor:
        """
        Converts the sampled value from the posterior distribution into a single size for the decoder.
        """
        return z

    def KLD(self, var_posterior: Distribution, prior: Distribution) -> storch.Tensor:
        """
        Computes the KL divergence between the variational posterior and the prior distribution
        """
        return torch.distributions.kl_divergence(var_posterior, prior).sum(-1)

    def name(self) -> str:
        return "vae"
