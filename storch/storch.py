from typing import List, Union

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
    return storch.reduce(torch.mean, dims=reduced_batches)(tensor, indices)


def sum(tensor: storch.Tensor, dims=List[Union[str, int]]) -> storch.Tensor:
    indices, reduced_batches = _convert_indices(tensor, dims)
    return storch.reduce(torch.sum, dims=reduced_batches)(tensor, indices)
