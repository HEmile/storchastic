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
from storch import sample, deterministic, cost, backward
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

    def KLD(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        # Here!
        KLD = self.KLD(mu, logvar)
        std = torch.exp(0.5 * logvar)
        dist = Normal(mu, std)
        # z = sample(dist, method=storch.method.ScoreFunction(), n=100)
        z = dist.sample()
        log_prob = dist.log_prob(z)
        return self.decode(z), KLD, z, log_prob.sum()


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCE = storch.nn.b_binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    print(BCE)
    return BCE


def train(epoch):
    model.train()
    storch.reset()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, KLD, z, log_prob = model(data)
        print(log_prob)
        BCE = loss_function(recon_batch, data)
        loss = BCE * log_prob + KLD
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

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
                comparison = storch.cat([data[:n],
                                      recon_batch.detach_tensor()[0].view(args.batch_size, 1, 28, 28)[:n]]) # Take the first sample (0)
                deterministic(save_image)(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

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