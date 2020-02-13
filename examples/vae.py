'''
Edited from https://github.com/pytorch/examples/blob/master/vae/main.py
'''

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from storch import deterministic, cost, backward
import storch
from torch.distributions import Normal

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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.method = storch.method.Reparameterization()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    @deterministic
    def decode(self, z):
        h3 = storch.relu(self.fc3(z))
        return self.fc4(h3).sigmoid()

    @cost
    def KLD(self, mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        # Here!
        KLD = self.KLD(mu, logvar)
        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)
        z = self.method(dist, n=5)
        return self.decode(z), KLD, z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
@cost
def loss_function(recon_x, x):
    BCE = storch.nn.b_binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    return BCE


def train(epoch):
    model.train()
    train_loss = 0
    storch.reset()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, KLD, z = model(data)
        BCE = loss_function(recon_batch, data)
        # storch.add_cost(BCE)
        cond_log = batch_idx % args.log_interval == 0
        cost, loss = backward(debug=False, accum_grads=cond_log)
        train_loss += loss.item()
        optimizer.step()
        z.total_expected_grad()
        if cond_log:
            grads_mean = []
            grads_std = []
            for i in range(10):
                optimizer.zero_grad()
                recon_batch, KLD, z = model(data)
                BCE = loss_function(recon_batch, data)
                storch.add_cost(BCE)
                cost, loss = backward()
                expected_grad = z.total_expected_grad()
                grads_mean.append(expected_grad[0].unsqueeze(0))
                grads_std.append(expected_grad[1].unsqueeze(0))
            def _var(t):
                m = torch.cat(t)
                mean = m.mean(0)
                squared_diff = (m - mean)**2
                sse = squared_diff.sum(0)
                return sse.mean()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCost: {:.6f}\tMean var {:.4E}\t Std var {:.4E}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data), cost.item() / len(data), _var(grads_mean), _var(grads_std)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, KLD, _ = model(data)
            test_loss += (loss_function(recon_batch, data).detach_tensor() + KLD.detach_tensor()).mean()
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
            im_sample = torch.randn(64, 20).to(device)
            im_sample = model.decode(im_sample).cpu()
            save_image(im_sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')