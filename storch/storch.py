from typing import List, Union, Optional

import storch
import torch


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
    if plates and plate_names:
        raise ValueError("Provide only one of plates and plate_names.")
    if not isinstance(tensor, storch.Tensor):
        if not plates:
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
    for plate in plates:
        if plate.n > 1:
            tensor = plate.reduce(tensor, detach_weights=detach_weights)
    return tensor
