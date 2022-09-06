from __future__ import annotations

from queue import Queue

import torch
import storch
from torch.distributions import Distribution
from collections import deque
from typing import List, Iterable, Any, Callable, Iterator, Dict, Tuple, Deque
import builtins
from itertools import product
from typing import Optional
from storch.exceptions import IllegalStorchExposeError
from storch.excluded_init import (
    exception_methods,
    excluded_methods,
    unwrap_only_methods,
    expand_methods,
)

# from storch.typing import BatchTensor


class Plate:
    def __init__(
        self,
        name: str,
        n: int,
        parents: List[Plate],
        weight: Optional[storch.Tensor] = None,
    ):
        self.weight = weight
        if weight is None:
            self.weight = torch.tensor(1.0 / n)
        self.name = name
        self.n = n
        self.parents = parents

    def __eq__(self, other) -> bool:
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
            if self.weight.shape != other.weight.shape:
                return False
            return self.weight._tensor.__eq__(other.weight._tensor).all()
        if isinstance(other.weight, storch.Tensor):
            return False
        # Neither of the weights are Tensors, so the weights must be equal as self.n==other.n
        return True

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
        if self.n == 1:
            return storch.reduce(lambda x: x * plate_weighting, self.name)(tensor)
        # Case: The weight is a single number. First sum, then multiply with the weight (usually taking the mean)
        elif plate_weighting.ndim == 0:
            return storch.sum(tensor, self) * plate_weighting

        # Case: There is a weight for each plate which is not dependent on the other batch dimensions
        elif plate_weighting.ndim == 1:
            index = tensor.get_plate_dim_index(self.name)
            plate_weighting = plate_weighting[
                (...,) + (None,) * (tensor.ndim - index - 1)
            ]
            weighted_tensor = tensor * plate_weighting
            return storch.sum(weighted_tensor, self)

        # Case: The weight is a vector of numbers equal to batch dimension. Assumes it is a storch.Tensor
        else:
            for parent_plate in self.parents:
                if parent_plate not in tensor.plates:
                    raise ValueError(
                        "Plate missing when reducing tensor: " + parent_plate.name
                    )
            weighted_tensor = tensor * plate_weighting
            return storch.sum(weighted_tensor, self)

    def on_collecting_args(self, plates: [Plate]) -> bool:
        """
        Gets called after a wrapper collected plates from its input arguments.

        Args:
            plates ([Plate]):  All collected plates

        Returns:
            bool: True if this plate should remain in the collected plates.

        """
        for plate in plates:
            if plate.name == self.name and plate != self:
                if not plate.on_duplicate_plate(self):
                    return False
        return True

    def on_duplicate_plate(self, plate: Plate) -> bool:
        raise ValueError("")

    def on_unwrap_tensor(self, tensor: storch.Tensor) -> storch.Tensor:
        """
        Gets called whenever the given tensor is being unwrapped and unsqueezed for batch use.

        Args:
            tensor (storch.Tensor): The input tensor that is being unwrapped

        Returns:
            storch.Tensor: The tensor that will be unwrapped and unsqueezed in the future. Can be a modification of the input tensor.

        """
        return tensor

    def index_in(self, plates: List[Plate]) -> int:
        return plates.index(self)

    def is_in(self, plates: Iterable[Plate]) -> bool:
        return self in plates


class Tensor:
    """
    A :class:`storch.Tensor` is a wrapper around a :class:`torch.Tensor` that acts like a normal :class:`torch.Tensor`
    with some restrictions and some extra data.

    By design, :class:`storch.Tensor` cannot expose the wrapped :class:`torch.Tensor` to regular Python control flow.
    For example, using a :class:`storch.Tensor` inside an ``if`` condition or in a ``for`` loop will throw an
    :class:`~storch.exceptions.IllegalStorchExposeError`. This is done because a node could be dependent on a Tensor that is used as
    a conditional to branch between different computation paths.  However, Python control flow will not register dependencies
    between nodes in the computation graph.

    The underlying :class:`torch.Tensor` can be unwrapped in two ways. The safe way is using the :func:`.deterministic`
    wrapper, which safely unwraps the :class:`storch.Tensor` and runs the function on the unwrapped :class:`torch.Tensor`.
    Note that all ``torch`` methods are automatically wrapped using :func:`.deterministic` when an input argument
    is :class:`storch.Tensor`.
    The unsafe way to unwrap the tensor is to access :attr:`storch.Tensor._tensor`. This should only be use when one is
    sure this will not introduce missing dependency links.

    Args:
        tensor (torch.Tensor): The tensor to wrap. The leftmost dimensions should correspond to the sizes of ``plates``
            that are larger than 1.
        parents ([storch.Tensor]): The parents of this Tensor. Parents represent the incoming links in stochastic
            computation graphs.
        plates ([storch.Plate]): The plates of this Tensor. Plates contain information about the sampling procedure and
            dependencies of this Tensor with respect to earlier samples.
        name (Optional[str]): The name of this Tensor.
    """

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
                    + ". This can be caused by different samples with the same name using a different amount of "
                    + "samples n or different weighting of the samples. Make sure that these samples use the same number of samples."
                )
            plate_names.add(plate.name)
            # plate length is 1. Ignore this dimension, as singleton dimensions should not exist.
            if plate.n == 1:
                continue
            if len(tensor.shape) <= batch_dims:
                raise ValueError(
                    "Got an input tensor with too few dimensions. We expected "
                    + str(len(plates))
                    + " plate dimensions. Instead, we found only "
                    + str(len(tensor.shape))
                    + " dimensions. Violated at dimension "
                    + str(batch_dims)
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
        self._parents: List[Tuple[Tensor, bool]] = []
        self._cleaned = False
        # DISCONTINUED FOR NOW BECAUSE OF PERFORMANCE OVERHEAD
        # differentiable_links = has_backwards_path(self, parents)
        for i, p in enumerate(parents):
            # TODO: Should I re-add this?
            # if p.is_cost:
            #     raise ValueError("Cost nodes cannot have children.")
            # TODO: DIFFERENTIABLE LINKS MANUALLY SET TO FALSE. THIS MIGHT CAUSE BUGS IN THE FUTURE
            self._parents.append((p, False))
            p._children.append((self, False))
        self._children = []
        self.plate_dims = batch_dims
        self.event_shape = tensor.shape[batch_dims:]
        self.event_dims = len(self.event_shape)
        self.plates = plates

    @classmethod
    def __torch_function__(cls, func: Callable, types, args=(), kwargs=None) -> Callable:
        """
        Called whenever a torch.* or torch.nn.functional.* method is being called on a storch.Tensor. This wraps
        that method in the deterministic wrapper to properly handle all input arguments and outputs.
        """
        if kwargs is None:
            kwargs = {}
        func_name = func.__name__
        if func_name in exception_methods:
            raise IllegalStorchExposeError(
                "Calling method " + func_name + " with storch tensors is not allowed."
            )
        if func_name in excluded_methods:
            return func(*args, **kwargs)

        if func_name in expand_methods:
            # Automatically expand empty plate dimensions. This is necessary for some loss functions, which
            # assume both inputs have exactly the same elements.
            return storch.wrappers._handle_deterministic(
                func, args, kwargs, expand_plates=True
            )
        # if func_name in unwrap_only_methods:
        #     return storch.wrappers._unpack_wrapper(func)(*args, *kwargs)

        return storch.wrappers._handle_deterministic(func, args, kwargs)

    def __getattr__(self, item) -> Any:
        """
        Called whenever an attribute is called on a storch.Tensor object that is not directly implemented by storch.Tensor.
        It defers it to the underlying torch.Tensor. If it is a callable (ie, torch.Tensor implements a function
        with the name item), it will wrap this callable with a deterministic wrapper.

        TODO: This should probably filter the methods
        """
        attr = getattr(torch.Tensor, item)
        if callable(attr):
            func_name = attr.__name__
            if func_name in exception_methods:
                raise IllegalStorchExposeError(
                    "Calling method "
                    + func_name
                    + " with storch tensors is not allowed."
                )
            if func_name in excluded_methods:
                return attr
            # if func_name in unwrap_only_methods:
            #     return storch.wrappers._unpack_wrapper(attr, self=self)
            return storch.wrappers._self_deterministic(attr, self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_sparse(self) -> bool:
        """
        Returns: True if the underlying tensor is sparse.
        """
        return self._tensor.is_sparse

    def __str__(self) -> str:
        t = (
            (self.name + ": " if self.name else "") + "Stochastic"
            if self.stochastic
            else ("Cost" if self.is_cost else "Deterministic")
        )
        return t + " " + str(self._tensor) + " Batch links: " + str(self.plates)

    def __repr__(self) -> str:
        return f"[{repr(self.name)}, {repr(self._tensor)}, {repr(self.plates)}"

    def __hash__(self) -> int:
        return object.__hash__(self)

    @storch.deterministic
    def __eq__(self, other) -> bool:
        return self.__eq__(other)

    @storch.deterministic(l_broadcast=False)
    def __getitem__(self, index):
        return self.__getitem__(index)

    @storch.deterministic(l_broadcast=False)
    def __setitem__(self, index, value):
        return self.__setitem__(index, value)

    def _walk_backwards(
        self,
        expand_fn: Callable[[Tensor], Iterator[Tuple[Tensor, bool]]],
        depth_first=True,
        reverse=False, # Only supported for breadth-first
        only_differentiable=False,
        repeat_visited=False,
        walk_fn=lambda x: x,
    ) -> Iterator[Tensor]:
        visited = set()
        visited_ordered = []
        if depth_first:
            S = [self]
            while S:
                v = S.pop()
                if repeat_visited or not v.is_in(visited):
                    yield walk_fn(v)
                    visited.add(v)
                    for w, d in expand_fn(v):
                        if d or not only_differentiable:
                            S.append(w)
        else:
            queue: Deque[Tensor] = deque()
            visited.add(self)
            queue.append(self)
            while queue:
                v = queue.popleft()
                if reverse:
                    visited_ordered.append(v)
                else:
                    yield walk_fn(v)
                for w, d in expand_fn(v):
                    if (repeat_visited or not w.is_in(visited)) and (
                        d or not only_differentiable
                    ):
                        visited.add(w)
                        queue.append(w)
            if reverse:
                for v in reversed(visited_ordered):
                    yield v

    def walk_parents(
        self,
        depth_first=True,
        reverse=False,
        only_differentiable=False,
        repeat_visited=False,
        walk_fn=lambda x: x,
    ) -> Iterator[Tensor]:
        """
        Searches through the parents of this Tensor in the stochastic computation graph.

        Args:
            depth_first: True to use depth first, otherwise breadth first.
            reverse: Reverse the order: If true, instead of first returning the immediate parents, return
                the parents furthest up, working towards the immediate parents.
                Currently only supported for breadth-first search.
            only_differentiable: True to only walk over edges that are differentiable
            repeat_visited:
            walk_fn: Optional function on :class:`storch.Tensor` that manipulates the nodes found.

        Returns:
            Iterator of type that is equal to the output type of ``walk_fn``.
        """
        return self._walk_backwards(
            lambda p: p._parents,
            depth_first,
            reverse,
            only_differentiable,
            repeat_visited,
            walk_fn,
        )

    def walk_children (
        self,
        depth_first=True,
        reverse=False,
        only_differentiable=False,
        repeat_visited=False,
        walk_fn=lambda x: x,
    ) -> Iterator[Tensor]:
        """
        Searches through the children of this Tensor in the stochastic computation graph.

        Args:
            depth_first: True to use depth first, otherwise breadth first.
            only_differentiable: True to only walk over edges that are differentiable
            repeat_visited:
            walk_fn: Optional function on :class:`storch.Tensor` that manipulates the nodes found.

        Returns:
            Iterator of type that is equal to the output type of ``walk_fn``.
        """
        return self._walk_backwards(
            lambda p: p._children,
            depth_first,
            reverse,
            only_differentiable,
            repeat_visited,
            walk_fn,
        )

    def _clean(self) -> None:
        """
        Cleans up :attr:`_children` and :attr:`_parents` for all nodes in the subgraph of this node (depth first)
        """
        if self._cleaned:
            return
        self._cleaned = True
        for (node, _) in self._children:
            node._clean()
        for (node, _) in self._parents:
            node._clean()
        self._children = []
        self._parents = []

    def detach_tensor(self) -> storch.Tensor:
        """
        Returns: A :class:`storch.Tensor` that is removed from PyTorch's differention graph.
            However, the tensor will remain present on the stochastic computation graph.
        """
        return self._tensor.detach()

    @property
    def stochastic(self) -> bool:
        """
        Returns:
            bool: True if this is a stochastic node in the stochastic computation graph, False otherwise.
        """
        return False

    @property
    def is_cost(self) -> bool:
        """
        Returns:
            bool: True if this is a cost node in the stochastic computation graph, False otherwise.
        """
        return False

    @property
    def parents(self) -> List[Tensor]:
        return list(map(lambda p: p[0], self._parents))

    @property
    def requires_grad(self) -> bool:
        return self._tensor.requires_grad

    @property
    def plate_shape(self) -> torch.Size:
        return self._tensor.shape[: self.plate_dims]

    def size(self, *args) -> torch.Size:
        return self._tensor.size(*args)

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

    @property
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

    def multi_dim_plates(self) -> List[Plate]:
        return list(filter(lambda p: p.n > 1, self.plates))

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

    def is_in(self, tensors: Iterable[Tensor]) -> bool:
        for tensor in tensors:
            if tensor is self:
                return True

        return False
    # region OperatorOverloads

    def __len__(self) -> int:
        return self._tensor.__len__()

    def __index__(self) -> int:
        raise IllegalStorchExposeError("Cannot use storch tensors as index.")

    @storch.deterministic
    def eq(self, other) -> bool:
        return self.eq(other)

    def __getstate__(self):
        raise NotImplementedError(
            "Pickle is currently not implemented for storch tensors."
        )

    def __setstate__(self, state):
        raise NotImplementedError(
            "Pickle is currently not implemented for storch tensors."
        )

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
        # TODO: This recognizes storch.Tensor as Iterable, even though it's not implemented.
        raise NotImplementedError("Cannot currently iterate over storch Tensors.")

    def detach_(self) -> Tensor:
        raise NotImplementedError("In place detach is not allowed on storch tensors.")

    def __add__(self, other):
        return torch.add(self, other)

    def __radd__(self, other):
        return torch.add(other, self)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def __pow__(self, other):
        return torch.pow(self, other)

    def __rpow__(self, other):
        return torch.pow(other, self)

    def __div__(self, other):
        return torch.div(self, other)

    def __rdiv__(self, other):
        return torch.div(other, self)

    def __mod__(self, other):
        return torch.remainder(self, other)

    def __rmod__(self, other):
        return torch.remainder(other, self)

    def __truediv__(self, other):
        return torch.true_divide(self, other)

    def __rtruediv__(self, other):
        return torch.true_divide(other, self)

    def __floordiv__(self, other):
        return torch.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return torch.floor_divide(other, self)

    def __abs__(self):
        return torch.abs(self)

    def __and__(self, other):
        return torch.logical_and(self, other)

    def __rand__(self, other):
        return torch.logical_and(other, self)

    def __ge__(self, other):
        return torch.ge(self, other)

    def __gt__(self, other):
        return torch.gt(self, other)

    @storch.deterministic
    def __invert__(self):
        return self.__invert__()

    def __le__(self, other):
        return torch.le(self, other)

    @storch.deterministic
    def __lshift__(self, other):
        return self.__lshift__(other)

    @storch.deterministic
    def __lshift__(self, other):
        return other.__lshift__(self)

    def __lt__(self, other):
        return torch.lt(self, other)

    def ne(self, other):
        return torch.ne(self, other)

    def __neg__(self):
        return torch.neg(self)

    def __or__(self, other):
        return torch.logical_or(self, other)

    def __ror__(self, other):
        return torch.logical_or(other, self)

    def __pos__(self):
        # TODO: Is this correct?
        return self

    @storch.deterministic
    def __rshift__(self, other):
        return self.__rshift__(other)

    @storch.deterministic
    def __rrshift__(self, other):
        return other.__rshift__(self)

    def __xor__(self, other):
        return torch.logical_xor(self, other)

    def __rxor__(self, other):
        return torch.logical_xor(other, self)

    # endregion




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
        plates.insert(0, Plate(plate_name, n, plates.copy(), weight))
        super().__init__(tensor, parents, plates, tensor_name)
        self.n = n

    # TODO: Should IndependentTensors be stochastic? Sometimes, like if it is denoting a minibatch, making them
    #  stochastic seems like it is correct. Are there other cases?
    def stochastic(self) -> bool:
        return True


class StochasticTensor(Tensor):
    """
    A :class:`storch.Tensor` that represents a stochastic node in the stochastic computation graph.

    Args:
        n (int): The size of the plate dimension created by this stochastic node.
        distribution: The distribution of this stochastic node.
        requires_grad (bool): True if we are interested in the gradients with respect to the parameters of the distribution of
            this stochastic node.
        
    """

    # TODO: Copy original tensor to make sure it cannot change using inplace
    def __init__(
        self,
        tensor: torch.Tensor,
        parents: [Tensor],
        plates: [Plate],
        name: str,
        n: int,
        distribution: Distribution,
        requires_grad: bool,
        method: Optional[storch.method.Method] = None,
    ):
        self.distribution: Distribution = distribution
        super().__init__(tensor, parents, plates, name)
        self._requires_grad = requires_grad
        self.n = n
        self.method = method
        self.param_grads = {}
        self._grad = None
        self._clean_hooks = []
        self._remove_handles = []

    @property
    def stochastic(self) -> bool:
        return True

    @property
    # TODO: Should not manually override it like this. The stochastic "requires_grad" should be a different method, so
    # that the meaning of requires_grad is consistent everywhere
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self) -> Dict[str, storch.Tensor]:
        return self.param_grads

    def _set_method(self, method: storch.method.Method):
        self.method = method

    def _clean(self) -> None:
        new_param_grads = {}
        for name, grad in self.param_grads.items():
            # In case higher-order derivatives are stored, remove these from the graph.
            new_param_grads[name] = grad.detach()
        for clean_hook in self._clean_hooks:
            clean_hook()
        for handle in self._remove_handles:
            handle.remove()
        self._clean_hooks = []
        self._remove_handles = []
        super()._clean()


is_tensor = lambda a: isinstance(a, torch.Tensor) or isinstance(a, Tensor)
from storch.util import has_backwards_path
