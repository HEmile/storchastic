from typing import List, Union, Optional, Tuple

import storch
from torch.distributions.utils import clamp_probs
import torch

from storch.typing import AnyTensor, _indices, _plates


def _convert_indices(tensor: storch.Tensor, dims: _indices) -> (List[int], List[str]):
    conv_indices = []
    red_batches = []
    if not isinstance(dims, List):
        dims = [dims]
    for index in dims:
        if isinstance(index, int):
            if index >= tensor.plate_dims or index < 0 and index >= -tensor.event_dims:
                conv_indices.append(index)
            else:
                print(tensor.shape, index)
                raise IndexError(
                    "Can only pass indexes for event dimensions."
                    + str(tensor)
                    + ". Index: "
                    + str(index)
                )
        else:
            if isinstance(index, storch.Plate):
                index = index.name
            conv_indices.append(tensor.get_plate_dim_index(index))
            red_batches.append(index)
    return tuple(conv_indices), red_batches


def mean(tensor: storch.Tensor, dims: _indices) -> storch.Tensor:
    """
    Simply takes the mean of the tensor over the dimensions given.
    WARNING: This does NOT weight the different elements according to the plates. You will very likely want to
    call the reduce_plates method instead.
    """
    indices, reduced_batches = _convert_indices(tensor, dims)
    return storch.reduce(torch.mean, plates=reduced_batches)(tensor, indices)


def sum(tensor: storch.Tensor, dims: _indices) -> storch.Tensor:
    """
    Simply sums the tensor over the dimensions given.
    WARNING: This does NOT weight the different elements according to the plates. You will very likely want to
    call the reduce_plates method instead.
    """
    indices, reduced_batches = _convert_indices(tensor, dims)
    return storch.reduce(torch.sum, plates=reduced_batches)(tensor, indices)


def logsumexp(tensor: storch.Tensor, dims: _indices) -> storch.Tensor:
    indices, reduced_batches = _convert_indices(tensor, dims)
    return storch.reduce(torch.logsumexp, plates=reduced_batches)(tensor, indices)


def expand_as(tensor: AnyTensor, expand_as: AnyTensor) -> AnyTensor:
    return storch.deterministic(torch.expand_as)(tensor, expand_as)


def _handle_inputs(
    tensor: AnyTensor, plates: Optional[_plates],
) -> (storch.Tensor, List[storch.Plate]):
    if isinstance(plates, storch.Plate):
        plates = [plates]
    if not isinstance(tensor, storch.Tensor):
        if not plates:
            raise ValueError("Make sure to pass plates when passing a torch.Tensor.")
        index_tensor = 0

        for plate in plates:
            if not isinstance(plate, storch.Plate):
                raise ValueError(
                    "Cannot handle plate names when passing a torch.Tensor"
                )
            if plate.n > 1:
                if tensor.shape[index_tensor] != plate.n:
                    raise ValueError(
                        "Received a tensor that does not align with the given plates."
                    )
                index_tensor += 1
        return storch.Tensor(tensor, [], plates), plates

    if not plates:
        return tensor, tensor.plates
    if isinstance(plates, str):
        return tensor, [tensor.get_plate(plates)]
    r_plates = []
    for plate in plates:
        if isinstance(plate, storch.Plate):
            r_plates.append(plate)
        else:
            r_plates.append(tensor.get_plate(plate))
    return tensor, r_plates


def gather(input: storch.Tensor, dim: str, index: storch.Tensor):
    # TODO: Should be allowed to accept int and storch.Plate as well
    return storch.deterministic(torch.gather, dim=dim, expand_plates=True)(
        input, index=index
    )


def reduce_plates(
    tensor: AnyTensor, plates: Optional[_plates] = None, detach_weights=True,
) -> storch.Tensor:
    """
    Reduce the tensor along the given plates. This takes into account how different samples are weighted, and should
    nearly always be used instead of reducing plate dimensions using the mean or the sum.
    By default, this reduces all plates.

    Args:
        tensor: Tensor to reduce
        plates: Plates to reduce. If None, this reduces all plates (default). Can be a string, Plate, or list of string
        and Plates.
        detach_weights: Whether to detach the weighting of the samples from the graph

    Returns:
        The reduced tensor
    """
    tensor, plates = _handle_inputs(tensor, plates)
    for plate in order_plates(plates, reverse=True):
        tensor = plate.reduce(tensor, detach_weights=detach_weights)

    return tensor


def _isroot(plate: storch.Plate, plates: [storch.Plate]):
    """
    Returns true if the plate is a root of the list of plates.
    """
    for parent in plate.parents:
        if parent in plates:
            return False
        if not _isroot(parent, plates):
            return False
    return True


def order_plates(plates: [storch.Plate], reverse=False):
    """
    Topologically order the given plates.
    Uses Kahn's algorithm.
    """
    sorted = []
    roots = []
    in_edges = {}
    out_edges = {p.name: [] for p in plates}
    for p in plates:
        if _isroot(p, plates):
            roots.append(p)
        in_edges[p.name] = p.parents.copy()
        for _p in p.parents:
            if _p.name in out_edges:
                out_edges[_p.name].append(p)
            # This is possible if the input list of plates does not contain the parent. We still need to register it
            # to make sure the list is sorted in global topological ordering!
            else:
                out_edges[_p.name] = []
    while roots:
        n = roots.pop()
        if n in plates:
            sorted.append(n)
        for m in out_edges[n.name]:
            remaining_edges = in_edges[m.name]
            remaining_edges.remove(n)
            if not remaining_edges:
                roots.append(m)
    for remaining_edges in in_edges.values():
        if remaining_edges and any(map(lambda p: p in plates, remaining_edges)):
            raise ValueError("List of plates contains a cycle")
    if reverse:
        return reversed(sorted)
    return sorted


def variance(
    tensor: AnyTensor,
    variance_plate: Union[storch.Plate, str],
    plates: Optional[_plates] = None,
    detach_weights=True,
) -> storch.Tensor:
    """
    Compute the variance of the tensor along the plate dimensions. This takes into account how different samples are weighted.

    Args:
        tensor: Tensor to compute variance over
        plates: Plates to reduce.
        detach_weights: Whether to detach the weighting of the samples from the graph
    Returns:
        The variance of the tensor.
    """
    tensor, plates = _handle_inputs(tensor, plates)
    found_plate = False
    for plate in plates:
        if plate == variance_plate:
            found_plate = True
            break
        if plate.name == variance_plate:
            variance_plate = plate
            found_plate = True
            break
    if not found_plate:
        raise ValueError(
            "Should pass a variance_plate that is included in the passed plates."
        )
    mean = variance_plate.reduce(tensor, detach_weights=detach_weights)
    variance = (tensor - mean) ** 2

    event_dims = tensor.event_dim_indices
    if len(event_dims) > 0:
        variance = variance.sum(tensor.event_dim_indices)
    return reduce_plates(variance, detach_weights=detach_weights)


def grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
) -> Tuple[storch.Tensor, ...]:
    """
    Helper method for computing torch.autograd.grad on storch tensors. Returns storch Tensors as well.
    """
    args, _, _, _ = storch.wrappers._prepare_args(
        [outputs, inputs, grad_outputs], {}, unwrap=True, align_tensors=False,
    )
    _outputs, _inputs, _grad_outputs = tuple(args)
    grads = torch.autograd.grad(
        _outputs,
        _inputs,
        grad_outputs=_grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        only_inputs=only_inputs,
        allow_unused=allow_unused,
    )
    storch_grad = []
    for i, grad in enumerate(grads):
        input = inputs[i]
        if isinstance(input, storch.Tensor):
            storch_grad.append(
                storch.Tensor(
                    grad, outputs + [input], input.plates, input.name + "_grad"
                )
            )
        else:
            storch_grad.append(storch.Tensor(grad, outputs, [], "grad"))
    return tuple(storch_grad)


@storch.deterministic(expand_plates=True)
def cat(*args, **kwargs):
    """
    Version of :func:`torch.cat` that is compatible with :class:`storch.Tensor`.
    Required because :meth:`torch.Tensor.__torch_function__` is not properly implemented for :func:`torch.cat`:
    https://github.com/pytorch/pytorch/issues/34294

    """
    return torch.cat(*args, **kwargs)


@storch.deterministic
def conditional_gumbel_rsample(
    hard_sample: torch.Tensor, probs: torch.Tensor, bernoulli: bool, temperature,
) -> torch.Tensor:
    """
    Conditionally re-samples from the distribution given the hard sample.
    This samples z \sim p(z|b), where b is the hard sample and p(z) is a gumbel distribution.
    """
    # Adapted from torch.distributions.relaxed_bernoulli and torch.distributions.relaxed_categorical
    shape = hard_sample.shape

    probs = clamp_probs(probs.expand_as(hard_sample))
    v = clamp_probs(torch.rand(shape, dtype=probs.dtype, device=probs.device))
    if bernoulli:
        pos_probs = probs[hard_sample == 1]
        v_prime = torch.zeros_like(hard_sample)
        # See https://arxiv.org/abs/1711.00123
        v_prime[hard_sample == 1] = v[hard_sample == 1] * pos_probs + (1 - pos_probs)
        v_prime[hard_sample == 0] = v[hard_sample == 0] * (1 - probs[hard_sample == 0])
        log_sample = (
            probs.log() + probs.log1p() + v_prime.log() + v_prime.log1p()
        ) / temperature
        return log_sample.sigmoid()
    # b=argmax(hard_sample)
    b = hard_sample.max(-1).indices
    # b = F.one_hot(b, hard_sample.shape[-1])

    # See https://arxiv.org/abs/1711.00123
    log_v = v.log()
    # i != b (indexing could maybe be improved here, but i doubt it'd be more efficient)
    log_v_b = torch.gather(log_v, -1, b.unsqueeze(-1))
    cond_gumbels = -(-torch.div(log_v, probs) - log_v_b).log()
    # i = b
    index_sample = hard_sample.bool()
    cond_gumbels[index_sample] = -(-log_v[index_sample]).log()
    scores = cond_gumbels / temperature
    return (scores - scores.logsumexp(dim=-1, keepdim=True)).exp()
