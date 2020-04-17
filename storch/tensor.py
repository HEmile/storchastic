from __future__ import annotations
import torch
import storch
from torch.distributions import Distribution
from collections import deque
from typing import Union, List, Tuple, Dict, Iterable, Any, Callable
import builtins
from itertools import product
from typing import Optional
from torch import Size
from storch.exceptions import IllegalStorchExposeError
from storch.excluded_init import (
    _exception_tensor,
    _unwrap_only_tensor,
    _excluded_tensor,
)

# from storch.typing import BatchTensor

_int = builtins.int
_size = Union[Size, List[_int], Tuple[_int, ...]]

_torch_dict = None


class Plate:
    def __init__(self, name: str, n: int, weight: Optional[storch.Tensor] = None):
        self.weight = weight
        if weight is None:
            self.weight = torch.tensor(1.0 / n)
        self.name = name
        self.n = n

    def __eq__(self, other):
        if not isinstance(other, Plate):
            return False
        if self.name != other.name:
            return False
        if self.n != other.n:
            # TODO: This should maybe return an error...?
            return False
        if isinstance(self.weight, storch.Tensor):
            if not isinstance(other.weight, storch.Tensor):
                return False
            if self.weight._tensor is other.weight._tensor:
                return True
            return self.weight._tensor.__eq__(other.weight._tensor).all()
        if isinstance(other.weight, storch.Tensor):
            return False
        return True  # Weights are equal if self.n == other.n

    def __str__(self):
        return self.name + ", " + str(self.n)

    def __repr__(self):
        return (
            "("
            + self.name.__repr__()
            + ", "
            + self.n.__repr__()
            + ", "
            + self.weight.__repr__()
            + ")"
        )

    def reduce(self, tensor: storch.Tensor, detach_weights=True):
        plate_weighting = self.weight
        if detach_weights:
            plate_weighting = self.weight.detach()
        # Case: The weight is a single number. First sum, then multiply with the weight (usually taking the mean)
        if plate_weighting.ndim == 0:
            return storch.sum(tensor, [self.name]) * plate_weighting

        # Case: There is a weight for each plate which is not dependent on the other batch dimensions
        elif plate_weighting.ndim == 1:
            index = tensor.get_plate_dim_index(self.name)
            plate_weighting = plate_weighting[
                (...,) + (None,) * (tensor.ndim - index - 1)
            ]
            weighted_tensor = tensor * plate_weighting
            return storch.sum(weighted_tensor, [self.name])

        # Case: The weight is a vector of numbers equal to batch dimension. Assumes it is a storch.Tensor
        else:
            weighted_tensor = tensor * plate_weighting
            return storch.sum(weighted_tensor, [self.name])

    def on_collecting_args(self, plates: [Plate]) -> bool:
        """
        Gets called after a wrapper collected plates from its input arguments.
        :param plates: All collected plates
        :return: Return True if this plate should remain in the collected plates.
        """
        return True

    def on_unwrap_tensor(self, tensor: storch.Tensor) -> storch.Tensor:
        """
        Gets called whenever the given tensor is being unwrapped and unsqueezed for batch use.
        :param tensor: The input tensor that is being unwrapped
        :return: The tensor that will be unwrapped and unsqueezed in the future. Can be a modification of the input tensor.
        """
        return tensor


class Tensor(torch.Tensor):
    def __init__(
        self,
        tensor: torch.Tensor,
        parents: [Tensor],
        plates: [Plate],
        name: Optional[str] = None,
    ):
        if isinstance(tensor, Tensor):
            raise TypeError(
                "storch.Tensors should be constructed with torch.Tensors, not other storch.Tensors."
            )
        plate_names = set()
        batch_dims = 0
        # Check whether this tensor does not violate the constraints imposed by the given batch_links
        for plate in plates:
            if plate.name in plate_names:
                raise ValueError(
                    "Plates contain two instances of same plate "
                    + plate.name
                    + ". This can be caused by different samples with the same name using a different amount of samples n or different weighting of the samples. Make sure that these samples use the same number of samples."
                )
            plate_names.add(plate.name)
            # plate length is 1. Ignore this dimension, as singleton dimensions should not exist.
            if plate.n == 1:
                continue
            if len(tensor.shape) <= batch_dims:
                raise ValueError(
                    "Got an input tensor with a shape too small for its surrounding batch. Violated at dimension "
                    + str(batch_dims)
                    + " and plate shape dimension "
                    + str(len(plates))
                    + ". Instead, it was "
                    + str(len(tensor.shape))
                )
            elif not tensor.shape[batch_dims] == plate.n:
                raise ValueError(
                    "Storch Tensors should take into account their surrounding plates. Violated at dimension "
                    + str(batch_dims)
                    + " and plate "
                    + plate.name
                    + " with size "
                    + str(plate.n)
                    + ". "
                    "Instead, it was "
                    + str(tensor.shape[batch_dims])
                    + ". Batch links: "
                    + str(plates)
                    + " Tensor shape: "
                    + str(tensor.shape)
                )
            batch_dims += 1

        self._name = name
        self._tensor = tensor
        self._parents = []
        for p in parents:
            # TODO: Should I re-add this?
            # if p.is_cost:
            #     raise ValueError("Cost nodes cannot have children.")
            differentiable_link = has_backwards_path(self, p)
            self._parents.append((p, differentiable_link))
            p._children.append((self, differentiable_link))
        self._children = []
        self.plate_dims = batch_dims
        self.event_shape = tensor.shape[batch_dims:]
        self.event_dims = len(self.event_shape)
        self.plates = plates

    @staticmethod
    def __new__(cls, *args, **kwargs):
        # Pass the input tensor to register this tensor in C. This will initialize an empty (0s?) tensor in the backend.
        # TODO: Does that mean it will require double the memory?
        # return super(Tensor, cls).__new__(cls, device=tensor.device)
        return super(Tensor, cls).__new__(cls)

    def __hash__(self):
        return object.__hash__(self)

    @property
    def name(self):
        return self._name

    @property
    def is_sparse(self):
        return self._tensor.is_sparse

    def __str__(self):
        t = (
            (self.name + ": " if self.name else "") + "Stochastic"
            if self.stochastic
            else ("Cost" if self.is_cost else "Deterministic")
        )
        return t + " " + str(self._tensor) + " Batch links: " + str(self.plates)

    def __repr__(self):
        return self._tensor.__repr__()

    @storch.deterministic
    def __eq__(self, other):
        return self.__eq__(other)

    def _walk(
        self,
        expand_fn,
        depth_first=True,
        only_differentiable=False,
        repeat_visited=False,
        walk_fn=lambda x: x,
    ):
        visited = set()
        if depth_first:
            S = [self]
            while S:
                v = S.pop()
                if repeat_visited or v not in visited:
                    yield walk_fn(v)
                    visited.add(v)
                    for w, d in expand_fn(v):
                        if d or not only_differentiable:
                            S.append(w)
        else:
            queue = deque()
            visited.add(self)
            queue.append(self)
            while queue:
                v = queue.popleft()
                yield walk_fn(v)
                for w, d in expand_fn(v):
                    if (repeat_visited or w not in visited) and (
                        d or not only_differentiable
                    ):
                        visited.add(w)
                        queue.append(w)

    def walk_parents(
        self,
        depth_first=True,
        only_differentiable=False,
        repeat_visited=False,
        walk_fn=lambda x: x,
    ):
        return self._walk(
            lambda p: p._parents,
            depth_first,
            only_differentiable,
            repeat_visited,
            walk_fn,
        )

    def walk_children(
        self,
        depth_first=True,
        only_differentiable=False,
        repeat_visited=False,
        walk_fn=lambda x: x,
    ):
        return self._walk(
            lambda p: p._children,
            depth_first,
            only_differentiable,
            repeat_visited,
            walk_fn,
        )

    def detach_tensor(self) -> torch.Tensor:
        return self._tensor.detach()

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def is_cost(self) -> bool:
        return False

    @property
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @property
    def plate_shape(self) -> torch.Size:
        return self._tensor.shape[: self.plate_dims]

    @property
    def shape(self) -> torch.Size:
        return self._tensor.size()

    def is_cuda(self):
        return self._tensor.is_cuda

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def layout(self):
        return self._tensor.layout

    @property
    def device(self):
        return self._tensor.device

    @property
    def grad(self):
        return self._tensor.grad

    def dim(self):
        return self._tensor.dim()

    def ndimension(self):
        return self._tensor.ndimension()

    @property
    def ndim(self):
        return self._tensor.ndim

    def register_hook(self, hook: Callable) -> Any:
        return self._tensor.register_hook(hook)

    def event_dim_indices(self):
        return range(self.plate_dims, self._tensor.dim())

    def get_plate(self, plate_name: str) -> Plate:
        for plate in self.plates:
            if plate.name == plate_name:
                return plate
        raise IndexError("Tensor has no such plate: " + plate_name + ".")

    def get_plate_dim_index(self, plate_name: str) -> int:
        for i, plate in enumerate(self.multi_dim_plates()):
            if plate.name == plate_name:
                return i
        raise IndexError(
            "Tensor has no such plate: "
            + plate_name
            + ". Alternatively, the dimension of this batch is 1."
        )

    def iterate_plate_indices(self) -> Iterable[List[int]]:
        ranges = list(map(lambda a: list(range(a)), self.plate_shape))
        return product(*ranges)

    def multi_dim_plates(self) -> Iterable[Plate]:
        for plate in self.plates:
            if plate.n > 1:
                yield plate

    def backward(
        self,
        gradient: Optional[Tensor] = None,
        keep_graph: bool = False,
        create_graph: bool = False,
        retain_graph: bool = False,
    ) -> None:
        raise NotImplementedError(
            "Cannot call .backward on storch.Tensor. Instead, register cost nodes using "
            "storch.add_cost, then use storch.backward()."
        )

    # region OperatorOverloads

    def __len__(self):
        return self._tensor.__len__()

    # TODO: Is this safe?
    def __index__(self):
        return self._tensor.__index__()

    # TODO: shouldn't this have @deterministic?
    def eq(self, other):
        return self.eq(other)

    def __getstate__(self):
        raise NotImplementedError(
            "Pickle is currently not implemented for storch tensors."
        )

    def __setstate__(self, state):
        raise NotImplementedError(
            "Pickle is currently not implemented for storch tensors."
        )

    def __and__(self, other):
        if isinstance(other, bool):
            raise IllegalStorchExposeError(
                "Calling 'and' with a bool exposes the underlying tensor as a bool."
            )
        return storch.deterministic(self._tensor.__and__)(other)

    def __bool__(self):
        raise IllegalStorchExposeError(
            "It is not allowed to convert storch tensors to boolean. Make sure to unwrap "
            "storch tensors to normal torch tensor to use this tensor as a boolean."
        )

    def __float__(self):
        raise IllegalStorchExposeError(
            "It is not allowed to convert storch tensors to float. Make sure to unwrap "
            "storch tensors to normal torch tensor to use this tensor as a float."
        )

    def __int__(self):
        raise IllegalStorchExposeError(
            "It is not allowed to convert storch tensors to int. Make sure to unwrap "
            "storch tensors to normal torch tensor to use this tensor as an int."
        )

    def __long__(self):
        raise IllegalStorchExposeError(
            "It is not allowed to convert storch tensors to long. Make sure to unwrap "
            "storch tensors to normal torch tensor to use this tensor as a long."
        )

    def __nonzero__(self) -> builtins.bool:
        raise IllegalStorchExposeError(
            "It is not allowed to convert storch tensors to boolean. Make sure to unwrap "
            "storch tensors to normal torch tensor to use this tensor as a boolean."
        )

    def __array__(self):
        self.numpy()

    def __array_wrap__(self):
        self.numpy()

    def numpy(self):
        raise IllegalStorchExposeError(
            "It is not allowed to convert storch tensors to numpy arrays. Make sure to unwrap "
            "storch tensors to normal torch tensor to use this tensor as a np.array."
        )

    def __contains__(self, item):
        raise IllegalStorchExposeError(
            "It is not allowed to expose storch tensors via in statements."
        )

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError(
            "There is currently no deep copying implementation for storch Tensors."
        )

    def __iter__(self):
        raise NotImplementedError("Cannot currently iterate over storch Tensors.")

    def detach_(self) -> Tensor:
        raise NotImplementedError("In place detach is not allowed on storch tensors.")

    # endregion


for m in dir(torch.Tensor):
    v = getattr(torch.Tensor, m)
    if (
        isinstance(v, Callable)
        and m not in Tensor.__dict__
        # and m not in object.__dict__
    ):
        if m in _exception_tensor:
            if storch._debug:
                print("exception:" + m)
            setattr(torch.Tensor, m, storch._exception_wrapper(v))
        elif m in _unwrap_only_tensor:
            if storch._debug:
                print("unwrap only:" + m)
            setattr(torch.Tensor, m, storch._unpack_wrapper(v))
        elif m not in _excluded_tensor:
            if storch._debug:
                print("in:" + m)
            setattr(torch.Tensor, m, storch.deterministic(v))
        elif storch._debug:
            print("excluded:" + m)
    elif storch._debug:
        print("out:" + m)

# Yes. This code is extremely weird. For some reason, when monkey patching torch.Tensor.__getitem__ and
# torch.Tensor.__setitem__, bizarre indexing bugs will happen that wouldn't happen when not monkey patching them.
# Unsqueezing the masking tensor sometimes seems to help...
# To do this, I also had to reset the torch.Tensor method to its original state. This bug should be fixed sometimes,
# as this is extremely messy code.
original_get = torch.Tensor.__getitem__

_getitem_level = 0


class WrapGet:
    def __init__(self, original_fn):
        self.original_fn = storch.deterministic(original_fn)

    def __get__(self, *args):
        # The __getitem__ method is a slotwrapper object that first calls the __get__ to create a method_wrapper.
        # This class is a hacky wrapper around those wrappers (pff) to allow compatibility with storch.
        self.is_arg_none = args[0] is None
        self.arg0 = args[0]
        return lambda *args: self.__call__(*args)

    def __call__(self, *args):
        # For some reason, when monkey patching __getitem__ and __setitem__, incorrect recursive calls happen that
        # cause wrong outputs. This wrapper seems to block some of these recursive calls, but this requires more
        # investigation as to why it works.
        global _getitem_level
        _getitem_level += 1
        # TODO: This seems to fix incorrect outcomes from wrapping. I still don't know why or how, though.
        try:
            if _getitem_level == 1 and not self.is_arg_none:
                out = self.original_fn(self.arg0, *args)
            else:
                out = self.original_fn(*args)
        except TypeError as e:
            if storch._debug:
                print(_getitem_level, e)
            if _getitem_level == 1:
                out = self.original_fn(self.arg0, *args)
        finally:
            _getitem_level -= 1

        return out


torch.Tensor.__getitem__ = WrapGet(torch.Tensor.__getitem__)

torch.Tensor.__setitem__ = WrapGet(torch.Tensor.__setitem__)


class CostTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, parents, plate_links: [Plate], name: str):
        super().__init__(tensor, parents, plate_links, name)

    @property
    def is_cost(self) -> bool:
        return True


class IndependentTensor(Tensor):
    """
    Used to denote independencies on a Tensor. This could for example be the minibatch dimension. The first dimension
    of the input tensor is taken to be independent and added as a batch dimension to the storch system.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        parents: [Tensor],
        plates: [Plate],
        tensor_name: str,
        plate_name: str,
        weight: Optional[storch.Tensor],
    ):
        n = tensor.shape[0]
        for plate in plates:
            if plate.name == plate_name:
                raise ValueError(
                    "Cannot create independent tensor with name "
                    + plate_name
                    + ". A parent sample has already used"
                    " this name. Use a different name for this independent dimension."
                )
        plates.insert(0, Plate(plate_name, n, weight))
        super().__init__(tensor, parents, plates, tensor_name)
        self.n = n

    # TODO: Should IndependentTensors be stochastic? Sometimes, like if it is denoting a minibatch, making them
    #  stochastic seems like it is correct. Are there other cases?
    def stochastic(self) -> bool:
        return True


class StochasticTensor(Tensor):
    # TODO: Copy original tensor to make sure it cannot change using inplace
    def __init__(
        self,
        tensor: torch.Tensor,
        parents: [Tensor],
        plates: [Plate],
        name: str,
        n: int,
        sampling_method: Optional[storch.Method],
        distribution: Distribution,
        requires_grad: bool,
    ):
        self.distribution = distribution
        super().__init__(tensor, parents, plates, name)
        self._requires_grad = requires_grad
        self.n = n
        self.sampling_method = sampling_method
        self.param_grads = {}
        self._grad = None

    @property
    def stochastic(self):
        return True

    @property
    # TODO: Should not manually override it like this. The stochastic "requires_grad" should be a different method, so
    # that the meaning of requires_grad is consistent everywhere
    def requires_grad(self):
        return self._requires_grad

    @property
    def grad(self):
        return self.param_grads


from storch.util import has_backwards_path
