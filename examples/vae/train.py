from __future__ import print_function
import argparse
from typing import Type

import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from storch import backward
import storch
from examples.dataloader.data_loader import data_loaders
from tensorboardX import SummaryWriter
from examples.vae.vae import VAE

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    bce = storch.nn.b_binary_cross_entropy(recon_x, x, reduction="sum")
    return bce


def train(epoch, model, train_loader, device, optimizer, args, writer):
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
            variances = {}
            if args.method != "expect":
                grads = {n: [] for n in z.grad}
                for i in range(10):
                    optimizer.zero_grad()
                    recon_batch, _, z = model(data)
                    storch.add_cost(loss_function(recon_batch, data), "reconstruction")
                    backward()
                    for n, grad in z.grad.items():
                        grads[n].append(grad.unsqueeze(0))
                variances = {}
                for n, gradz in grads.items():
                    variances[n] = torch.cat(gradz).var(0).mean()

            step = 100.0 * batch_idx / len(train_loader)
            global_step = 100 * (epoch - 1) + step
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}\tAdditive Loss: {:.6f}\t Logits var {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    step,
                    cost.item(),
                    (loss - cost).item(),
                    variances,
                )
            )
            writer.add_scalar("train/ELBO", cost, global_step)
            writer.add_scalar("train/loss", loss, global_step)
            for n, var in variances.items():
                writer.add_scalar("train/variance/" + n, var, global_step)
    avg_train_loss = train_loss / (batch_idx + 1)
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_train_loss))
    return avg_train_loss


def test(epoch, model, test_loader, device):
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


def main(vae: Type[VAE]):
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
        default="none",
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

    model = vae(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_train_loss = 1000.0
    best_test_loss = 1000.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch, model, train_loader, device, optimizer, args, writer)
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        test_loss = test(epoch, model, test_loader, device)
        writer.add_scalar("test_loss", test_loss, 100 * epoch)
        writer.flush()
        if test_loss < best_test_loss:
            best_test_loss = test_loss
        with torch.no_grad():
            im_sample = model.prior([args.latents]).sample((64,))
            im_sample = model.decode(im_sample).cpu()
            save_image(
                im_sample.view(64, 1, 28, 28), "results/sample_" + str(epoch) + ".png",
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