"""
Edited from https://github.com/pytorch/examples/blob/master/vae/main.py
Reproduce experiments from Kool 2020 and Yin 2019
"""

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from storch import deterministic, backward
import storch
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
from examples.dataloader.data_loader import data_loaders
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="VAE MNIST Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--method",
    type=str,
    default="gumbel",
    help="Method in {gumbel, gumbel_straight, score}",
)
parser.add_argument(
    "--baseline",
    type=str,
    default="batch_average",
    help="What baseline to use for the score function.",
)
parser.add_argument(
    "--latents",
    type=int,
    default=20,
    help="How many latent variables with 10 categories to use",
)
parser.add_argument(
    "--samples", type=int, default=1, help="How large of a budget to use"
)
parser.add_argument("--dataset", type=str, default="fixedMNIST")
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

writer = SummaryWriter(comment="discrete_vae_" + args.method)
print(args)
writer.add_text("hyperparameters", str(args), global_step=0)

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

train_loader, test_loader = data_loaders(args)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        if args.method == "gumbel":
            self.sampling_method = storch.method.GumbelSoftmax()
        elif args.method == "gumbel_straight":
            self.sampling_method = storch.method.GumbelSoftmax(straight_through=True)
        elif args.method == "score":
            self.sampling_method = storch.method.ScoreFunction(
                baseline_factory=args.baseline
            )
        elif args.method == "expect":
            self.sampling_method = storch.method.Expect()
        self.latents = args.latents
        self.samples = args.samples
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latents * 10)
        self.fc4 = nn.Linear(self.latents * 10, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.activation = lambda x: F.leaky_relu(x, negative_slope=0.1)

    # @deterministic
    def encode(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        return self.fc3(h2)

    # @deterministic
    def decode(self, z):
        h3 = self.activation(self.fc4(z))
        h4 = self.activation(self.fc5(h3))
        return self.fc6(h4).sigmoid()

    def KLD(self, p, q):
        kld = torch.distributions.kl_divergence(p, q).sum(-1)
        storch.add_cost(kld, "KL-divergence")
        return kld

    def forward(self, x):
        logits = self.encode(x)
        logits = logits.reshape(logits.shape[:-1] + (self.latents, 10))

        q = OneHotCategorical(logits=logits)
        p = OneHotCategorical(probs=torch.ones_like(logits) / (1.0 / 10.0))
        KLD = self.KLD(q, p)
        z = self.sampling_method("z", q, n=self.samples)
        zp = z.reshape(z.shape[:-2] + (self.latents * 10,))
        return self.decode(zp), KLD, z


model = VAE(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    bce = storch.nn.b_binary_cross_entropy(recon_x, x, reduction="sum")
    return bce


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        storch.reset()

        # Denote the minibatch dimension as being independent
        data = storch.denote_independent(data.view(-1, 784), 0, "data")
        recon_batch, KLD, z = model(data)
        storch.add_cost(loss_function(recon_batch, data), "reconstruction")
        cond_log = batch_idx % args.log_interval == 0
        cost, loss = backward()
        train_loss += cost.item()

        optimizer.step()
        if cond_log:
            # Variance of expect method is 0 by definition.
            if args.method == "expect":
                variance = 0.0
            else:
                grads_logits = []
                for i in range(10):
                    optimizer.zero_grad()
                    recon_batch, _, z = model(data)
                    storch.add_cost(loss_function(recon_batch, data), "reconstruction")
                    backward()
                    grads_logits.append(z.grad["logits"].unsqueeze(0))

                variance = torch.cat(grads_logits).var(0).mean()

            step = 100.0 * batch_idx / len(train_loader)
            global_step = 100 * (epoch - 1) + step
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}\tAdditive Loss: {:.6f}\t Logits var {:.4E}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    step,
                    cost.item(),
                    (loss - cost).item(),
                    variance,
                )
            )
            writer.add_scalar("train/ELBO", cost, global_step)
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/variance", variance, global_step)
    avg_train_loss = train_loss / (batch_idx + 1)
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_train_loss))
    return avg_train_loss


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data = storch.denote_independent(data.view(-1, 784), 0, "data")
            recon_batch, KLD, _ = model(data)
            test_loss += (
                loss_function(recon_batch, data).detach_tensor()
            ).mean() + KLD.detach_tensor().mean()
            if i == 0:
                n = min(data.size(0), 8)
                # comparison = storch.cat([data[:n],
                #                       recon_batch.detach_tensor()[0].view(args.batch_size, 1, 28, 28)[:n]]) # Take the first sample (0)
                # deterministic(save_image)(comparison.cpu(),
                #          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= i + 1
    print("====> Test set loss: {:.4f}".format(test_loss))
    return test_loss


if __name__ == "__main__":
    best_train_loss = 1000.0
    best_test_loss = 1000.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        test_loss = test(epoch)
        writer.add_scalar("test_loss", test_loss, 100 * epoch)
        writer.flush()
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        with torch.no_grad():
            im_sample = torch.randn(64, args.latents * 10).to(device)
            im_sample = model.decode(im_sample).cpu()
            save_image(
                im_sample.view(64, 1, 28, 28), "results/sample_" + str(epoch) + ".png"
            )
    measures = {
        "hparams/best_train_loss": best_train_loss,
        "hparams/best_test_loss": best_test_loss,
        # "hparams/train_loss": train_loss, "hparams/test_loss": test_loss,
        "train/loss": train_loss,
        "test_loss": test_loss,
    }
    writer.add_hparams(vars(args), measures, global_step=100 * epoch)
writer.close()
