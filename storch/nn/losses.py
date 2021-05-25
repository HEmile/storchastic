from typing import Union, List

import storch
import torch
from storch import deterministic
import torch.nn.functional as F


@deterministic(unwrap=False)
def b_binary_cross_entropy(
    input: storch.Tensor,
    target: storch.Tensor,
    dims: Union[str, List[str]] = None,
    weight=None,
    reduction: str = "mean",
):
    r"""Function that measures the Binary Cross Entropy in a batched way
    between the target and the output.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
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

    if not dims:
        dims = []
    if isinstance(dims, str):
        dims = [dims]

    target = target.expand_as(input)
    unreduced = deterministic(F.binary_cross_entropy)(
        input, target, weight, reduction="none"
    )

    # unreduced = _loss(input, target, weight)
    indices = list(unreduced.event_dim_indices) + dims

    if reduction == "mean":
        return storch.mean(unreduced, indices)
    elif reduction == "sum":
        return storch.sum(unreduced, indices)
    elif reduction == "none":
        return unreduced
