'''
Edited from https://github.com/pytorch/examples/blob/master/vae/main.py
Reproduce experiments from Kool 2020 and Yin 2019
'''

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from storch import deterministic, cost, backward
import storch
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical
from examples.dataloader.data_loader import data_loaders

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument("--method", type=str, default="gumbel", help="Method in {gumbel, gumbel_straight, score}")
parser.add_argument("--baseline", type=str, default="batch_average", help="What baseline to use for the score function.")
parser.add_argument("--latents", type=int, default=20, help="How many latent variables with 10 categories to use")
parser.add_argument("--samples", type=int, default=1, help="How large of a budget to use")
parser.add_argument("--dataset", type=str, default="fixedMNIST")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

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
            self.sampling_method = storch.method.ScoreFunction(baseline_factory=args.baseline)
        self.latents = args.latents
        self.samples = args.samples
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latents * 10)
        self.fc4 = nn.Linear(self.latents * 10, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

        self.activation = lambda x: F.leaky_relu(x, negative_slope=0.1)

    def encode(self, x):
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        return self.fc3(h2)

    @deterministic
    def decode(self, z):
        h3 = self.activation(self.fc4(z))
        h4 = self.activation(self.fc5(h3))
        return self.fc6(h4).sigmoid()

    @cost
    def KLD(self, p, q):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        div = torch.distributions.kl_divergence(p, q)
        return div.sum()

    def forward(self, x):
        logits = self.encode(x.view(-1, 784))
        logits = logits.reshape(logits.shape[:-1] + (self.latents, 10))
        q = OneHotCategorical(logits=logits)
        p = OneHotCategorical(probs=torch.ones_like(logits) / (1./10.))
        KLD = self.KLD(q, p)
        z = self.sampling_method("z", q, n=self.samples)
        zp = z.reshape(z.shape[:-2] + (self.latents * 10,))
        return self.decode(zp), KLD, z


model = VAE(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
@cost
def loss_function(recon_x, x):
    BCE = storch.nn.b_binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        storch.reset()

        recon_batch, KLD, z = model(data)
        loss_function(recon_batch, data)
        cond_log = batch_idx % args.log_interval == 0
        cost, loss = backward()
        train_loss += loss.item()

        optimizer.step()
        if cond_log:
            grads_logits = []
            for i in range(10):
                optimizer.zero_grad()
                recon_batch, _, z = model(data)
                loss_function(recon_batch, data)
                backward()
                expected_grad = z.total_expected_grad()
                grads_logits.append(expected_grad["logits"].unsqueeze(0))
            def _var(t):
                m = torch.cat(t)
                mean = m.mean(0)
                squared_diff = (m - mean)**2
                sse = squared_diff.sum(0)
                return sse.mean()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCost: {:.6f}\t Logits var {:.4E}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), cost.item() / len(data), _var(grads_logits)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, KLD, _ = model(data)
            test_loss += (loss_function(recon_batch, data).detach_tensor()).mean()
            if i == 0:
                n = min(data.size(0), 8)
                # comparison = storch.cat([data[:n],
                #                       recon_batch.detach_tensor()[0].view(args.batch_size, 1, 28, 28)[:n]]) # Take the first sample (0)
                # deterministic(save_image)(comparison.cpu(),
                #          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):

        train(epoch)
        test(epoch)
        with torch.no_grad():
            im_sample = torch.randn(64, args.latents * 10).to(device)
            im_sample = model.decode(im_sample).cpu()
            save_image(im_sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')


