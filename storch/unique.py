from typing import Optional

import storch
import torch

"""
This module removes all plates from the tensor, and computes functions over the unique elements. Using clever bookkeeping,
we can retrieve the associated plates to retrieve the original plates.
This is useful for (sequences of) discrete output distributions that sample with replacement. We can greatly reduce
the computation of the deterministic functions (such as a deterministic reward) by only computing the function over
the unique elements.

Note: this code is highly experimental.
"""


class UniquePlate(storch.Plate):
    def __init__(
        self,
        name: str,
        amt_unique: int,
        shrunken_plates: [storch.Plate],
        inv_indexing: torch.Tensor,
    ):
        super().__init__(name, amt_unique, [])
        self.inv_indexing = inv_indexing
        self.shrunken_plates = shrunken_plates
        self.weight = None

    def reduce(self, unique_tensor: storch.Tensor, detach_weights=True):
        non_unique = self.undo_unique(unique_tensor)
        return storch.reduce_plates(
            non_unique, plates=self.shrunken_plates, detach_weights=detach_weights
        )

    def on_collecting_args(self, plates: [storch.Plate]) -> bool:
        for plate in plates:
            if self.has_shrunken(plate):
                raise ValueError(
                    "Cannot call deterministic wrapper with input tensors of which some are unique and "
                    "some are not. Use storch.undo_unique on the unique tensors before calling."
                )
        return True

    def has_shrunken(self, plate: storch.Plate) -> bool:
        for _plate in self.shrunken_plates:
            if plate == _plate:
                return True
            if isinstance(_plate, UniquePlate) and _plate.has_shrunken(plate):
                return True
        return False

    def undo_unique(self, unique_tensor: storch.Tensor) -> torch.Tensor:
        """
        Convert the unique tensor back to the non-unique format, then add the old plates back in
        # TODO: Make sure self.shrunken_plates is added
        # TODO: What if unique_tensor contains new plates after the unique?
        :param unique_tensor:
        :return:
        """
        plate_idx = unique_tensor.get_plate_dim_index(self.name)
        with storch.ignore_wrapping():
            dim_swapped = unique_tensor.transpose(
                plate_idx, unique_tensor.plate_dims - 1
            )
            fl_selected = torch.index_select(
                dim_swapped, dim=0, index=self.inv_indexing
            )
            selected = fl_selected.reshape(
                tuple(map(lambda p: p.n, self.shrunken_plates)) + fl_selected.shape[1:]
            )
            return storch.Tensor(
                selected,
                [unique_tensor],
                self.shrunken_plates + unique_tensor.plates,
                "undo_unique_" + unique_tensor.name,
            )


def unique(tensor: storch.Tensor, event_dim: Optional[int] = 0) -> storch.Tensor:
    with storch.ignore_wrapping():
        fl_tensor = torch.flatten(tensor, tensor.plate_dims)
        uniq, inverse_indexing = torch.unique(
            fl_tensor, return_inverse=True, dim=event_dim
        )
    inverse_indexing = storch.Tensor(
        inverse_indexing, [tensor], tensor.plates, "inv_index_" + tensor.name
    )
    uq_plate = UniquePlate(
        "uq_plate_" + tensor.name,
        uniq.shape[0],
        tensor.multi_dim_plates(),
        inverse_indexing,
    )
    return storch.Tensor(uniq, [tensor], [uq_plate], "unique_" + tensor.name)


def undo_unique(tensor: storch.Tensor) -> storch.Tensor:
    while is_unique(tensor):
        for plate in tensor.plates:
            if isinstance(plate, UniquePlate):
                tensor = plate.undo_unique(tensor)
    return tensor


def is_unique(tensor: storch.Tensor) -> bool:
    for plate in tensor.plates:
        if isinstance(plate, UniquePlate):
            return True
    return False
