"""
Adapted from https://github.com/yoonholee/pytorch-vae/blob/master/data_loader/data_loader.py
"""

import torch
from torchvision import transforms, datasets
from .fixed_mnist import fixedMNIST


def data_loaders(args):
    if args.dataset == "normalMNIST":
        kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.data_dir,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args.data_dir, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
    elif args.dataset == "fixedMNIST":
        loader_fn, root = fixedMNIST, args.data_dir + "/fixedmnist"
        kwargs = {"num_workers": 4, "pin_memory": True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(
            loader_fn(root, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        test_loader = torch.utils.data.DataLoader(  # need test bs <=64 to make L_5000 tractable in one pass
            loader_fn(
                root, train=False, download=True, transform=transforms.ToTensor()
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
    return train_loader, test_loader
