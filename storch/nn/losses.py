import storch
import torch
from storch import deterministic
from torch.nn import _reduction as _Reduction
from torch._C import _infer_size
from storch.typing import AnyTensor, Dims
from typing import Optional


def b_binary_cross_entropy(input: AnyTensor, target: torch.Tensor, weight=None, reduction: str = 'mean'):
    r"""Function that measures the Binary Cross Entropy in a batched way
    between the target and the output.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> input = torch.randn((3, 2), requires_grad=True)
        >>> target = torch.rand((3, 2), requires_grad=False)
        >>> loss = b_binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """
    if isinstance(input, storch.Tensor):
        indices = input.event_dim_indices()
        if target.size() != input.event_shape:
            raise ValueError("Using a target size ({}) that is different to the input size ({}). "
                             "Please ensure they have the same size.".format(target.size(), input.event_shape))
    else:
        offset_dim = input.dim() - target.dim()
        indices = list(range(input.dim() - target.dim()))
        for i in range(target.dim()):
            if input.shape[offset_dim + i] != target.shape[i]:
                raise ValueError("Input and target are invalid for broadcasting.")


    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)
    else:
        weight = torch.ones_like(target)

    @deterministic
    def _loss(input):
        epsilon = 1e-6
        return -weight * (target * (input + epsilon).log() + (1. - target) * (1. - input + epsilon).log())

    unreduced = _loss(input)
    if reduction == "mean":
        return unreduced.mean(dim=indices)
    elif reduction == "sum":
        return unreduced.sum(dim=indices)
    elif reduction == "none":
        return unreduced
