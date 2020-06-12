import torch
import torch.nn as nn
import storch
from torch.distributions import OneHotCategorical

torch.manual_seed(0)


class DiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2 * 10)
        self.fc4 = nn.Linear(2 * 10, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x):
        h1 = self.fc1(x).relu()
        h2 = self.fc2(h1).relu()
        return self.fc3(h2)

    def decode(self, z):
        h3 = self.fc4(z).relu()
        h4 = self.fc5(h3).relu()
        return self.fc6(h4).sigmoid()


def generative_story(
    method: storch.method.Method, model: DiscreteVAE, data: torch.Tensor
):
    x = storch.denote_independent(data.view(-1, 784), 0, "data")

    # Encode data. Shape: (data, 2 * 10)
    q_logits = model.encode(x)
    # Shape: (data, 2, 10)
    q_logits = q_logits.reshape(-1, 2, 10)
    q = OneHotCategorical(probs=q_logits.softmax(dim=-1))
    # Sample from variational posterior
    z = method(q)

    prior = OneHotCategorical(probs=torch.ones_like(q.probs) / 10.0)
    # Shape: (data)
    KL_div = torch.distributions.kl_divergence(q, prior).sum(-1)
    storch.add_cost(KL_div, "kl-div")

    z_in = z.reshape(z.shape[:-2] + (2 * 10,))
    reconstruction = model.decode(z_in)
    bce = torch.nn.BCELoss(reduction="none")(reconstruction, x).sum(-1)
    # bce = torch.nn.BCELoss(reduction="sum")(reconstruction, x)
    storch.add_cost(bce, "reconstruction")
    return z


from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor(),
    ),
    shuffle=True,
    batch_size=64,
)


def train(method: storch.method.Method, train_loader):
    model = DiscreteVAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(5):
        print("Epoch:" + str(epoch + 1))
        for i, (data, _) in enumerate(train_loader):
            # if i % 300 == 0:
            # evaluate(method, model, data, optimizer)
            optimizer.zero_grad()

            generative_story(method, model, data)
            elbo = storch.backward()
            optimizer.step()
            if i % 300 == 0:
                print("Training ELBO " + str(elbo.item()))


def evaluate(method: storch.method.Method, model: DiscreteVAE, data, optimizer):
    # Compute expected gradient
    optimizer.zero_grad()

    z = generative_story(storch.method.Expect("z"), model, data)
    storch.backward()
    expected_gradient = z.param_grads["probs"]

    # Collect gradient samples
    gradients = []
    for i in range(100):
        optimizer.zero_grad()

        z = generative_story(method, model, data)
        elbo = storch.backward()
        gradients.append(z.param_grads["probs"])

    gradients = storch.gather_samples(gradients, "gradients")
    mean_gradient = storch.reduce_plates(gradients, "gradients")
    bias_gradient = (
        storch.reduce_plates((mean_gradient - expected_gradient) ** 2)
    ).sum()
    print(
        "Training ELBO "
        + str(elbo.item())
        + " Gradient variance "
        + str(storch.variance(gradients, "gradients")._tensor.item())
        + " Gradient bias "
        + str(bias_gradient._tensor.item())
    )


#
# train(
#     storch.method.ScoreFunction("z", n_samples=10, baseline_factory="batch_average"),
#     train_loader,
# )


train(
    storch.method.Expect("z"), train_loader,
)
