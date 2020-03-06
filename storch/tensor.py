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
from storch.excluded_init import _exception_tensor, _unwrap_only_tensor, _excluded_tensor

# from storch.typing import BatchTensor

_int = builtins.int
_size = Union[Size, List[_int], Tuple[_int, ...]]

_torch_dict = None

Plate = Tuple[str, int]

class Tensor(torch.Tensor):
    def __init__(self, tensor: torch.Tensor, parents: [Tensor],
                 batch_links: [Plate], name: Optional[str] = None):
        if isinstance(tensor, Tensor):
            raise TypeError("storch.Tensors should be constructed with torch.Tensors, not other storch.Tensors.")
        plate_names = set()
        batch_dims = 0
        # Check whether this tensor does not violate the constraints imposed by the given batch_links
        for plate in batch_links:
            plate_name, plate_n = plate
            if plate_name in plate_names:
                raise ValueError("Batch links contain two instances of plate " + plate_name + ". This can be caused by "
                                 "different samples with the same name using a different amount of samples n. "
                                 "Make sure that these samples use the same number of samples.")
            plate_names.add(plate_name)
            if plate_n == 1:  # plate length is 1. Ignore this dimension, as singleton dimensions should not exist.
                continue
            if len(tensor.shape) <= batch_dims:
                raise ValueError(
                    "Got an input tensor with a shape too small for its surrounding batch. Violated at dimension "
                    + str(batch_dims) + " and plate shape dimension " + str(len(batch_links)) + ". Instead, it was " + str(
                        len(tensor.shape)))
            elif not tensor.shape[batch_dims] == plate_n:
                raise ValueError(
                    "Storch Tensors should take into account their surrounding plates. Violated at dimension " + str(batch_dims)
                    + " and plate " + plate_name + " with size " + str(plate_n) + ". "
                    "Instead, it was " + str(tensor.shape[batch_dims]) + ". Batch links: " + str(batch_links) + " Tensor shape: "
                    + str(tensor.shape))
            batch_dims += 1

        self._name = name
        self._tensor = tensor
        self._parents = []
        for p in parents:
            if p.is_cost:
                raise ValueError("Cost nodes cannot have children.")
            differentiable_link = has_backwards_path(self, p)
            self._parents.append((p, differentiable_link))
            p._children.append((self, differentiable_link))
        self._children = []
        self.batch_dims = batch_dims
        self.event_shape = tensor.shape[batch_dims:]
        self.event_dims = len(self.event_shape)
        self.batch_links = batch_links

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
        t = (self.name + ": " if self.name else "") + \
            "Stochastic" if self.stochastic else ("Cost" if self.is_cost else "Deterministic")
        return t + " " + str(self._tensor) + " Batch links: " + str(self.batch_links)

    def __repr__(self):
        return object.__repr__(self)

    @storch.deterministic
    def __eq__(self, other):
        return self.__eq__(other)

    def _walk(self, expand_fn, depth_first=True, only_differentiable=False, repeat_visited=False, walk_fn=lambda x: x):
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
                    if (repeat_visited or w not in visited) and (d or not only_differentiable):
                        visited.add(w)
                        queue.append(w)

    def walk_parents(self, depth_first=True, only_differentiable=False, repeat_visited=False, walk_fn=lambda x: x):
        return self._walk(lambda p: p._parents, depth_first, only_differentiable, repeat_visited, walk_fn)

    def walk_children(self, depth_first=True, only_differentiable=False, repeat_visited=False, walk_fn=lambda x: x):
        return self._walk(lambda p: p._children, depth_first, only_differentiable, repeat_visited, walk_fn)

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
    def batch_shape(self) -> torch.Size:
        return self._tensor.shape[:self.batch_dims]

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

    def register_hook(self, hook: Callable) -> Any:
        return self._tensor.register_hook(hook)

    def event_dim_indices(self):
        return range(self.batch_dims, self._tensor.dim())

    def batch_dim_indices(self):
        return range(self.batch_dims)

    def get_batch_dim_index(self, batch_name: str):
        for i, (plate_name, _) in enumerate(self.multi_dim_plates()):
            if plate_name == batch_name:
                return i
        raise IndexError("Tensor has no such batch: " + batch_name + ". Alternatively, the dimension of this batch is 1.")

    def iterate_batch_indices(self):
        ranges = list(map(lambda a: list(range(a)), self.batch_shape))
        return product(*ranges)

    def multi_dim_plates(self) -> Iterable[Plate]:
        for plate_name, plate_n in self.batch_links:
            if plate_n > 1:
                yield plate_name, plate_n

    def backward(self, gradient: Optional[Tensor]=None, keep_graph: bool=False, create_graph: bool=False) -> None:
        raise NotImplementedError("Cannot call .backward on storch.Tensor. Instead, register cost nodes using "
                                  "storch.add_cost, then use storch.backward().")


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
        raise NotImplementedError("Pickle is currently not implemented for storch tensors.")

    def __setstate__(self, state):
        raise NotImplementedError("Pickle is currently not implemented for storch tensors.")

    def __and__(self, other):
        if isinstance(other, bool):
            raise IllegalStorchExposeError("Calling 'and' with a bool exposes the underlying tensor as a bool.")
        return storch.deterministic(self._tensor.__and__)(other)

    def __bool__(self):
        raise IllegalStorchExposeError("It is not allowed to convert storch tensors to boolean. Make sure to unwrap "
                                      "storch tensors to normal torch tensor to use this tensor as a boolean.")

    def __float__(self):
        raise IllegalStorchExposeError("It is not allowed to convert storch tensors to float. Make sure to unwrap "
                                      "storch tensors to normal torch tensor to use this tensor as a float.")

    def __int__(self):
        raise IllegalStorchExposeError("It is not allowed to convert storch tensors to int. Make sure to unwrap "
                                      "storch tensors to normal torch tensor to use this tensor as an int.")

    def __long__(self):
        raise IllegalStorchExposeError("It is not allowed to convert storch tensors to long. Make sure to unwrap "
                                      "storch tensors to normal torch tensor to use this tensor as a long.")

    def __nonzero__(self) -> builtins.bool:
        raise IllegalStorchExposeError("It is not allowed to convert storch tensors to boolean. Make sure to unwrap "
                                       "storch tensors to normal torch tensor to use this tensor as a boolean.")

    def __array__(self):
        self.numpy()

    def __array_wrap__(self):
        self.numpy()

    def numpy(self):
        raise IllegalStorchExposeError(
            "It is not allowed to convert storch tensors to numpy arrays. Make sure to unwrap "
            "storch tensors to normal torch tensor to use this tensor as a np.array.")

    def __contains__(self, item):
        raise IllegalStorchExposeError("It is not allowed to expose storch tensors via in statements.")

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError("There is currently no deep copying implementation for storch Tensors.")

    def __iter__(self):
        raise NotImplementedError("Cannot currently iterate over storch Tensors.")

    def detach_(self) -> Tensor:
        raise NotImplementedError("In place detach is not allowed on storch tensors.")
    # endregion


for m in dir(torch.Tensor):
    v = getattr(torch.Tensor, m)
    if isinstance(v, Callable) and m not in Tensor.__dict__ and m not in object.__dict__:
        if m in _exception_tensor:
            setattr(torch.Tensor, m, storch._exception_wrapper(v))
        elif m in _unwrap_only_tensor:
            setattr(torch.Tensor, m , storch._unpack_wrapper(v))
        elif m not in _excluded_tensor:
            setattr(torch.Tensor, m, storch.deterministic(v))

# Yes. This code is extremely weird. For some reason, when monkey patching torch.Tensor.__getitem__ and
# torch.Tensor.__setitem__, bizarre indexing bugs will happen that wouldn't happen when not monkey patching them.
# Unsqueezing the masking tensor sometimes seems to help...
# To do this, I also had to reset the torch.Tensor method to its original state. This bug should be fixed sometimes,
# as this is extremely messy code.
original_get = torch.Tensor.__getitem__
def wrap_get(a, b):
    try:
        return storch.deterministic(original_get)(a, b)
    except IndexError:
        if storch._debug:
            print("Got indexing error at __getitem__. Trying to resolve this using the unsqueeze hack.")

        @storch.deterministic
        def _weird_wrap(a, b):
            if isinstance(b, torch.Tensor):
                b = b.unsqueeze(0)
            return original_get(a, b)

        torch.Tensor.__getitem__ = original_get
        o = _weird_wrap(a, b)
        torch.Tensor.__getitem__ = wrap_get
        return o
torch.Tensor.__getitem__ = wrap_get

original_set = torch.Tensor.__setitem__
def wrap_set(a, b, v):
    try:
        return storch.deterministic(original_set)(a, b, v)
    except IndexError:
        if storch._debug:
            print("Got indexing error at __setitem__. Trying to resolve this using the unsqueeze hack.")

        @storch.deterministic
        def _weird_wrap(a, b, v):
            if isinstance(b, torch.Tensor):
                b = b.unsqueeze(0)
            return original_set(a, b, v)

        torch.Tensor.__setitem__ = original_set
        o = _weird_wrap(a, b, v)
        torch.Tensor.__setitem__ = wrap_set
        return o
torch.Tensor.__setitem__ = wrap_set


class CostTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, parents, batch_links: [Tuple[str, int]], name):
        super().__init__(tensor, parents, batch_links, name)

    @property
    def is_cost(self) -> bool:
        return True


class IndependentTensor(Tensor):
    """
    Used to denote independencies on a Tensor. This could for example be the minibatch dimension. The first dimension
    of the input tensor is taken to be independent and added as a batch dimension to the storch system.
    """

    def __init__(self, tensor: torch.Tensor, parents: [Tensor],
                 batch_links: [Tuple[str, int]], name: str):
        n = tensor.shape[0]
        for plate_name, plate_n in batch_links:
            if plate_name == name:
                raise ValueError(
                    "Cannot create independent tensor with name " + name + ". A parent sample has already used"
                    " this name. Use a different name for this independent dimension.")
        batch_links.insert(0, (name, n))
        super().__init__(tensor, parents, batch_links, name)
        self.n = n

    # TODO: Should IndependentTensors be stochastic? Sometimes, like if it is denoting a minibatch, making them
    #  stochastic seems like it is correct. Are there other cases?
    def stochastic(self) -> bool:
        return True


class StochasticTensor(Tensor):
    # TODO: Copy original tensor to make sure it cannot change using inplace
    def __init__(self, tensor: torch.Tensor, parents: [Tensor], sampling_method: storch.Method,
                 batch_links: [Tuple[str, int]],
                 distribution: Distribution, requires_grad: bool, n: int, name: str):
        for plate_name, plate_n in batch_links:
            if plate_name == name:
                raise ValueError("Cannot create stochastic tensor with name " + name + ". A parent sample has already used"
                                " this name. Use a different name for this sample.")
        batch_links.insert(0, (name, n))
        self.n = n
        self.distribution = distribution
        super().__init__(tensor, parents, batch_links, name)
        self.sampling_method = sampling_method
        self._requires_grad = requires_grad
        self._accum_grads = {}
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
        return self._accum_grads

    def total_expected_grad(self) -> Dict[str, torch.Tensor]:
        r = {}
        indices = self.batch_dim_indices()
        for name, grad in self._accum_grads.items():
            tensor = getattr(self.distribution, name)
            if grad.dim() == tensor.dim():
                r[name] = grad
            else:
                r[name] = grad.mean(dim=indices)
        return r

    def total_variance_grad(self) -> Dict[str, torch.Tensor]:
        """
        Computes the total variance on the gradient of the parameters of this distribution over all simulations .
        :return:
        """
        r = {}
        indices = self.batch_dim_indices()
        for name, grad in self._accum_grads.items():
            tensor = getattr(self.distribution, name)
            if grad.dim() == tensor.dim():
                raise ValueError("There are no batched dimensions to take statistics over. Make sure to call backwards "
                                 "with accum_grad=True")
            expected = grad.mean(dim=indices)
            diff = grad - expected
            squared_diff = diff * diff
            sse = squared_diff.sum(dim=indices)
            r[name] = sse.mean()
        return r


from storch.util import has_backwards_path
