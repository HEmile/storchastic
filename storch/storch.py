from typing import List, Union, Optional, Tuple

import storch
import torch

from storch import deterministic


def _convert_indices(
    tensor: storch.Tensor, dims=List[Union[str, int]]
) -> (List[int], List[str]):
    conv_indices = []
    red_batches = []
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
            conv_indices.append(tensor.get_plate_dim_index(index))
            red_batches.append(index)
    return tuple(conv_indices), red_batches


def mean(tensor: storch.Tensor, dims=List[Union[str, int]]) -> storch.Tensor:
    indices, reduced_batches = _convert_indices(tensor, dims)
    return storch.reduce(torch.mean, plates=reduced_batches)(tensor, indices)


def sum(tensor: storch.Tensor, dims=List[Union[str, int]]) -> storch.Tensor:
    indices, reduced_batches = _convert_indices(tensor, dims)
    return storch.reduce(torch.sum, plates=reduced_batches)(tensor, indices)


def _handle_inputs(
    tensor: torch.Tensor,
    plates: Optional[List[storch.Plate]],
    plate_names: Optional[List[str]],
) -> (storch.Tensor, List[storch.Plate]):
    if plates and plate_names:
        raise ValueError("Provide only one of plates and plate_names.")
    if not isinstance(tensor, storch.Tensor):
        if not plates or plate_names:
            raise ValueError("Make sure to pass plates when passing a torch.Tensor.")
        index_tensor = 0
        for plate in plates:
            if plate.n > 1:
                if tensor.shape[index_tensor] != plate.n:
                    raise ValueError(
                        "Received a tensor that does not align with the given plates."
                    )
                index_tensor += 1
        tensor = storch.Tensor(tensor, [], plates)
    else:
        if plate_names:
            plates = []
            for plate in tensor.plates:
                if plate.name in plate_names:
                    plates.append(plate)
        elif not plates:
            plates = tensor.plates
    return tensor, plates


def reduce_plates(
    tensor: torch.Tensor,
    *,
    plates: Optional[List[storch.Plate]] = None,
    plate_names: Optional[List[str]] = None,
    detach_weights=True,
) -> storch.Tensor:
    """
    Reduce the tensor along the given plates. This takes into account how different samples are weighted, and should
    nearly always be used instead of reducing plate dimensions using the mean or the sum.
    :param tensor: Tensor to reduce
    :param plates: Plates to reduce. Cannot be used together with plate_names
    :param plate_names: Names of plates to reduce. Cannot be used togeter with plates
    :param detach_weights: Whether to detach the weighting of the samples from the graph
    :return: The reduced tensor
    """
    tensor, plates = _handle_inputs(tensor, plates, plate_names)
    for plate in plates:
        if plate.n > 1:
            tensor = plate.reduce(tensor, detach_weights=detach_weights)
    return tensor


def variance(
    tensor: torch.Tensor,
    variance_plate: Union[storch.Plate, str],
    *,
    plates: Optional[List[storch.Plate]] = None,
    plate_names: Optional[List[str]] = None,
    detach_weights=True,
) -> storch.Tensor:
    """
    Compute the variance of the tensor along the plate dimensions. This takes into account how different samples are weighted.
    :param tensor: Tensor to compute variance over
    :param plates: Plates to reduce. Cannot be used together with plate_names
    :param plate_names: Names of plates to reduce. Cannot be used togeter with plates
    :param detach_weights: Whether to detach the weighting of the samples from the graph
    :return: The reduced tensor
    """
    tensor, plates = _handle_inputs(tensor, plates, plate_names)
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
    variance = ((tensor - mean) ** 2).sum(tensor.event_dim_indices())
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
    args, _, _, _ = storch.wrappers._handle_args(
        True, False, outputs, inputs, grad_outputs
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
