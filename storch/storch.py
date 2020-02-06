from storch import deterministic
import torch


@deterministic
def cat(tensors, dim=0):
    return torch.cat(tensors, dim)
