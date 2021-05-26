from __future__ import print_function
import argparse
from typing import Type

import torch
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from examples.dataloader.data_loader import data_loaders
from storch import backward
from storch.method import Expect
import storch
from torch.utils.tensorboard import SummaryWriter
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
        cost = backward()
        train_loss += cost.item()

        optimizer.step()

        cond_log = batch_idx % args.log_interval == 0

        if cond_log:
            step = 100.0 * batch_idx / len(train_loader)
            global_step = 100 * (epoch - 1) + step

            # Variance of expect method is 0 by definition.
            variances = {}
            if args.method != "expect" and args.variance_samples > 1:
                _consider_param = "probs"
                if args.latents < 3:
                    old_method = model.sampling_method
                    model.sampling_method = Expect("z")
                    optimizer.zero_grad()
                    recon_batch, _, z = model(data)
                    storch.add_cost(loss_function(recon_batch, data), "reconstruction")
                    backward()
                    expect_grad = storch.reduce_plates(
                        z.grad[_consider_param]
                    ).detach_tensor()

                    optimizer.zero_grad()
                    model.sampling_method = old_method
                grads = {n: [] for n in z.grad}

                for i in range(args.variance_samples):
                    optimizer.zero_grad()
                    recon_batch, _, z = model(data)
                    storch.add_cost(loss_function(recon_batch, data), "reconstruction")
                    backward()

                    for param_name, grad in z.grad.items():
                        # Make sure to reduce the data dimension and detach, for memory reasons.
                        grads[param_name].append(
                            storch.reduce_plates(grad).detach_tensor()
                        )

                variances = {}
                for param_name, gradz in grads.items():
                    # Create a new independent dimension for the different gradient samples
                    grad_samples = storch.gather_samples(gradz, "variance")
                    # Compute the variance over this independent dimension
                    variances[param_name] = storch.variance(
                        grad_samples, "variance"
                    )._tensor
                    if param_name == _consider_param and args.latents < 3:
                        mean = storch.reduce_plates(grad_samples, "variance")
                        mse = storch.reduce_plates(
                            (grad_samples - expect_grad) ** 2
                        ).sum()
                        bias = (storch.reduce_plates((mean - expect_grad) ** 2)).sum()
                        print("mse", mse._tensor.item())
                        # Should approach 0 when increasing variance_samples for unbiased estimators.
                        print("bias", bias._tensor.item())
                        writer.add_scalar("train/probs_bias", bias._tensor, global_step)
                        writer.add_scalar("train/probs_mse", mse._tensor, global_step)

            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tCost: {:.6f}\t Logits var {}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    step,
                    cost.item(),
                    variances,
                )
            )
            writer.add_scalar("train/ELBO", cost, global_step)
            for param_name, var in variances.items():
                writer.add_scalar("train/variance/" + param_name, var, global_step)
    avg_train_loss = train_loss / (batch_idx + 1)
    print("====> Epoch: {} Average loss: {:.4f}".format(epoch, avg_train_loss))
    return avg_train_loss


def test(epoch: int, model: torch.nn.Module, test_loader, device):
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
            storch.reset()

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
    parser.add_argument(
        "--variance_samples",
        type=int,
        default=10,
        help="How many samples to use to compute the variance of the estimators.",
    )
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="fixedMNIST")
    parser.add_argument("--out_dir", type=str, default="./outputs/")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    train_loader, test_loader = data_loaders(args)

    model = vae(args).to(device)
    writer = SummaryWriter(args.out_dir + model.name() + "_" + args.method)
    print(args)
    writer.add_text("hyperparameters", str(args), global_step=0)

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
        # TODO: Sampling currently doesn't work because of model.prior requiring the posterior.
        # with torch.no_grad():
        #     im_sample = model.prior([args.latents]).sample((64,))
        #     im_sample = model.decode(im_sample).cpu()
        #     save_image(
        #         im_sample.view(64, 1, 28, 28), args.out_dir + "results/sample_" + str(epoch) + ".png",
        #     )
    measures = {
        "hparams/best_train_loss": best_train_loss,
        "hparams/best_test_loss": best_test_loss,
        # "hparams/train_loss": train_loss, "hparams/test_loss": test_loss,
        "train/loss": train_loss,
        "test_loss": test_loss,
    }
    writer.add_hparams(vars(args), measures)
    writer.close()
