Discrete Variational Autoencoder
================================
    .. code-block:: python

        import torch
        import torch.nn as nn
        import storch
        from storch.method import ScoreFunction

        class DiscreteVAE(nn.Module):
            def __init__(self):
                self.method = ScoreFunction("z", 8, baseline_factory="batch_average")
                self.fc1 = nn.Linear(784, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 20 * 10)
                self.fc4 = nn.Linear(20 * 10, 256)
                self.fc5 = nn.Linear(256, 512)
                self.fc6 = nn.Linear(512, 784)

            def encode(self, x):
                h1 = self.fc1(x).relu()
                h2 = self.fc2(h1).relu()
                return self.fc3(h2).reshape(logits.shape[:-1] + (20, 10))

            def decode(self, z):
                z = z.reshape(z.shape[:-2] + (20 * 10,))
                h3 = self.fc4(z).relu()
                h4 = self.fc5(h3).relu()
                return self.fc6(h4).sigmoid()

            def KLD(self, q):
                p = torch.distributions.OneHotCategorical(probs=torch.ones_like(q.logits) / (1.0 / 10.0))
                return torch.distributions.kl_divergence(p, q).sum(-1)

            def forward(self, x):
                q = torch.distributions.OneHotCategorical(logits=self.encode(x))
                KLD = self.KLD(q)
                z = self.method("z", q, n=8)
                return self.decode(z), KLD

        model = DiscreteVAE()
        for data in minibatches():
            optimizer.zero_grad()
            # Denote the minibatch dimension as being independent
            data = storch.denote_independent(data.view(-1, 784), 0, "data")

            # Compute the output of the model
            recon_batch, KLD = model(data)

            # Register the two cost functions
            storch.add_cost(KLD)
            storch.add_cost(storch.nn.b_binary_cross_entropy(recon_batch, data, reduction="sum"))

            # Go backward through both deterministic and stochastic nodes
            average_ELBO, _ = storch.backward()
            print(average_ELBO)

            optimizer.step()


    .. code-block:: python

        import torch
        import storch
        from vae import minibatches, encode, decode, KLD

        method = storch.method.ScoreFunction("z", 8, baseline_factory="batch_average")
        for data in minibatches():
            optimizer.zero_grad()
            # Denote the minibatch dimension as being independent
            data = storch.denote_independent(data.view(-1, 784), 0, "data")

            # Define the variational distribution given the data, and sample latent variables
            q = torch.distributions.OneHotCategorical(logits=encode(data))
            z = method(q)

            # Compute and register the KL divergence and reconstruction losses to form the ELBO
            reconstruction = decode(z)
            storch.add_cost(KLD(q))
            storch.add_cost(storch.nn.b_binary_cross_entropy(reconstruction, data, reduction="sum"))

            # Go backward through both deterministic and stochastic nodes, and optimize
            average_ELBO, _ = storch.backward()
            optimizer.step()

    .. code-block:: python

        import torch
        import storch
        from vae import minibatches, encode, decode, KLD

        method = ScoreFunctionLOO("z", 8, baseline="batch_average")
        for data in minibatches():
            optimizer.zero_grad()
            # Denote the minibatch dimension as being independent
            data = storch.denote_independent(data.view(-1, 784), 0, "data")

            # Define variational distribution given data, and sample latent variables
            q = torch.distributions.OneHotCategorical(logits=encode(data))
            z = method(q)

            # Compute and register the KL divergence and reconstruction losses to form the ELBO
            reconstruction = decode(z)
            storch.add_cost(KLD(q))
            storch.add_cost(storch.nn.b_binary_cross_entropy(reconstruction, data))

            # Backward pass through deterministic and stochastic nodes, and optimize
            ELBO = storch.backward()
            optimizer.step()

    .. code-block:: python
    class ScoreFunctionLOO(Method):
        def proposal_dist(self, distr: Distribution, amt_samples: int,
        ) -> torch.Tensor:
            return distr.sample((amt_samples,))

        def estimator(self, tensor: StochasticTensor, cost: CostTensor
        ) -> Tuple[Optional[storch.Tensor], Optional[storch.Tensor]]:
            # Compute the gradient function (
            log_prob = tensor.distribution.log_prob(tensor)
            sum_costs = storch.sum(costs.detach(), tensor.name)
            baseline = (sum_costs - costs) / (tensor.n - 1)
            return log_prob, (1.0 - log_prob) * baseline