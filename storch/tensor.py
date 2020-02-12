from __future__ import annotations
import torch
import storch
from torch.distributions import Distribution
from collections import deque
from typing import Union, List, Tuple, Dict
import builtins
from itertools import product

_int = builtins.int

class Tensor:

    def __init__(self, tensor: torch.Tensor, parents: [Tensor], batch_links: [StochasticTensor]):
        for i, plate in enumerate(batch_links):
            if len(tensor.shape) <= i:
                raise ValueError(
                "Got an input tensor with a shape too small for its surrounding batch. Violated at dimension "
                + str(i) + " and plate shape dimension " + str(len(batch_links)) + ". Instead, it was " + str(
                    len(tensor.shape)))
            elif not tensor.shape[i] == plate.n:
                raise ValueError(
                    "Storch Tensors should take into account their surrounding plates. Violated at dimension " + str(i)
                    + " and plate size " + str(plate.n) + ". Instead, it was " + str(tensor.shape[i]))


        self._tensor = tensor
        self._parents = []
        for p in parents:
            if p.is_cost:
                raise ValueError("Cost nodes cannot have children.")
            differentiable_link = has_backwards_path(self, p)
            self._parents.append((p, differentiable_link))
            p._children.append((self, differentiable_link))
        self._children = []
        self.event_shape = tensor.shape[len(batch_links):]
        self.batch_links = batch_links

    def __str__(self):
        t = "Stochastic" if self.stochastic else ("Cost" if self.is_cost else "Deterministic")
        return t + " " + str(self._tensor.shape)

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

    def walk_parents(self, depth_first=True, only_differentiable=False, repeat_visited=False, walk_fn=lambda x:x):
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
        return torch.Size(map(lambda s: s.n, self.batch_links))

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def grad(self):
        return self._tensor.grad

    def dim(self):
        return self._tensor.dim()

    def event_dim_indices(self):
        return list(range(len(self.batch_links), self._tensor.dim()))

    def batch_dim_indices(self):
        return list(range(len(self.batch_links)))

    def iterate_batch_indices(self):
        ranges = list(map(lambda a: list(range(a)), self.batch_shape))
        return product(*ranges)

    # region UnwrappedFunctions
    # The reason all these functions work and don't go into infinite recursion is because they unwraps
    # the self input storch.Tensor, so that the self now is the unwrapped node._tensor

    # region OperatorOverloads
    @storch.deterministic
    def __add__(self, other):
        return self.__add__(other)

    @storch.deterministic
    def __radd__(self, other):
        return self.__radd__(other)

    @storch.deterministic
    def __sub__(self, other):
        return self.__sub__(other)

    @storch.deterministic
    def __mul__(self, other):
        return self.__mul__(other)

    @storch.deterministic
    def __rmul__(self, other):
        return self.__rmul__(other)

    @storch.deterministic
    def __matmul__(self, other):
        return self.__matmul__(other)

    @storch.deterministic
    def __pow__(self, other):
        return self.__pow__(other)

    @storch.deterministic
    def __div__(self, other):
        return self.__div__(other)

    @storch.deterministic
    def __mod__(self, other):
        return self.__mod__(other)

    @storch.deterministic
    def __truediv__(self, other):
        return self.__truediv__(other)

    @storch.deterministic
    def __floordiv__(self, other):
        return self.__floordiv__(other)

    @storch.deterministic
    def __rfloordiv__(self, other):
        return self.__rfloordiv__(other)

    @storch.deterministic
    def __abs__(self):
        return self.__abs__()

    @storch.deterministic
    def __and__(self, other):
        return self.__and__(other)

    def eq(self, other):
        return self.eq(other)

    @storch.deterministic
    def __ge__(self, other):
        return self.__ge__(other)

    @storch.deterministic
    def __getitem__(self, indices: Union[None, _int, slice, Tensor, List, Tuple]):
        # TODO: properly test this
        return self.__getitem__(indices)

    @storch.deterministic
    def __gt__(self, other):
        return self.__gt__(other)

    @storch.deterministic
    def __invert__(self):
        return self.__invert__()

    @storch.deterministic
    def __le__(self, other):
        return self.__le__(other)

    @storch.deterministic
    def __lshift__(self, other):
        return self.__lshift__(other)

    @storch.deterministic
    def __lt__(self, other):
        return self.__lt__(other)

    @storch.deterministic
    def ne(self, other):
        return self.ne(other)

    @storch.deterministic
    def __neg__(self):
        return self.__neg__()

    @storch.deterministic
    def __or__(self, other):
        return self.__or__(other)

    @storch.deterministic
    def __rshift__(self, other):
        return self.__rshift__(other)

    @storch.deterministic
    def __xor__(self, other):
        return self.__xor__(other)
    #endregion

    # region AlphabeticUnwraps
    @property
    @storch.deterministic
    def T(self):
        return self.T

    def is_cuda(self):
        return self._tensor.is_cuda

    def device(self):
        return self._tensor.device

    @storch.deterministic
    def abs(self):
        return self.abs()

    @storch.deterministic
    def acos(self):
        return self.acos()

    @storch.deterministic
    def addbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return self.addbmm(batch1, batch2, beta=beta, alpha=alpha)

    @storch.deterministic
    def addcdiv(self, batch1, batch2, *, value=1):
        return self.addcdiv(batch1, batch2, value=value)

    @storch.deterministic
    def addcmul(self, batch1, batch2, *, value=1):
        return self.addcmul(batch1, batch2, value=value)

    @storch.deterministic
    def addmm(self, batch1, batch2, *, beta=1, alpha=1):
        return self.addmm(batch1, batch2, beta=beta, alpha=alpha)

    @storch.deterministic
    def addmv(self, batch1, batch2, *, beta=1, alpha=1):
        return self.addmv(batch1, batch2, beta=beta, alpha=alpha)

    @storch.deterministic
    def addr(self, batch1, batch2, *, beta=1, alpha=1):
        return self.addr(batch1, batch2, beta=beta, alpha=alpha)

    @storch.deterministic
    def angle(self):
        return self.angle()

    @storch.deterministic
    def argmax(self, dim=None, keepdim=False):
        if dim:
            return self.argmax(dim, keepdim)
        return self.argmax()

    @storch.deterministic
    def argmin(self, dim=None, keepdim=False):
        if dim:
            return self.argmin(dim, keepdim)
        return self.argmin()

    @storch.deterministic
    def argsort(self, dim=None, keepdim=False):
        if dim:
            return self.argsort(dim, keepdim)
        return self.argsort()

    @storch.deterministic
    def asin(self):
        return self.asin()

    @storch.deterministic
    def as_strided(self, size, stride, storage_offset=0):
        return self.as_strided(size, stride, storage_offset)

    @storch.deterministic
    def atan(self):
        return self.atan()

    @storch.deterministic
    def atan2(self):
        return self.atan2()

    @storch.deterministic
    def baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
        return self.baddbmm(batch1, batch2, beta=beta, alpha=alpha)

    @storch.deterministic
    def bincount(self, weights=None, minlength=0):
        if weights:
            return self.bincount(weights, minlength)
        return self.bincount(minlength=minlength)

    @storch.deterministic
    def bitwise_not(self):
        return self.bitwise_not()

    @storch.deterministic
    def bitwise_xor(self):
        return self.bitwise_xor()

    @storch.deterministic
    def bmm(self, batch2):
        return self.bmm(batch2)

    @storch.deterministic
    def ceil(self):
        return self.ceil()

    @storch.deterministic
    def cholesky(self, upper=False):
        return self.cholesky(upper)

    @storch.deterministic
    def cholesky_inverse(self, upper=False):
        return self.cholesky_inverse(upper)

    @storch.deterministic
    def cholesky_solve(self, input2, upper=False):
        return self.cholesky_solve(input2, upper)

    @storch.deterministic
    def chunk(self, chunks, dim=0):
        return self.chunk(chunks, dim)

    @storch.deterministic
    def clamp(self, a, b):
        return self.clamp(a, b)

    @storch.deterministic
    def clone(self):
        return self.clone()

    @storch.deterministic
    def contiguous(self):
        return self.contiguous()

    @storch.deterministic
    def conj(self):
        return self.conj()

    @storch.deterministic
    def cos(self):
        return self.cos()

    @storch.deterministic
    def cosh(self):
        return self.cosh()

    @storch.deterministic
    def cpu(self):
        return self.cpu()

    @storch.deterministic
    def cross(self, other, dim=-1):
        return self.cross(other, dim)

    @storch.deterministic
    def cuda(self, device=None, non_blocking=False):
        if device:
            return self.cuda(device, non_blocking)
        return self.cuda(non_blocking=non_blocking)

    @storch.deterministic
    def cumprod(self, dim, dtype=None):
        return self.cumprod(dim, dtype)

    @storch.deterministic
    def cumsum(self, dim, dtype=None):
        return self.cumsum(dim, dtype)

    @storch.deterministic
    def dequantize(self):
        return self.dequantize()

    @storch.deterministic
    def det(self):
        return self.det()

    @storch.deterministic
    def detach(self):
        return self.detach()

    @storch.deterministic
    def diag(self, diagonal=0):
        return self.diag(diagonal)

    @storch.deterministic
    def diag_embed(self, offset=0, dim1=-2, dim2=-1):
        return self.diag_embed(offset, dim1, dim2)

    @storch.deterministic
    def diag_flat(self, offset=0):
        return self.diag_flat(offset)

    @storch.deterministic
    def diagonal(self, offset=0, dim1=-2, dim2=-1):
        return self.diagonal(offset, dim1, dim2)

    @storch.deterministic
    def digamma(self):
        return self.digamma()

    @storch.deterministic
    def dist(self, other, p=2):
        return self.dist(other, p)

    @storch.deterministic
    def dot(self, other):
        return self.dot(other)

    @storch.deterministic
    def eig(self, eigenvectors=False):
        return self.eig(eigenvectors)

    def element_size(self):
        return self._tensor.element_size()

    @storch.deterministic
    def erf(self):
        return self.erf()

    @storch.deterministic
    def erfc(self):
        return self.erfc()

    @storch.deterministic
    def erfinv(self):
        return self.erfinv()

    @storch.deterministic
    def exp(self):
        return self.exp()

    @storch.deterministic
    def expm1(self):
        return self.expm1()

    @storch.deterministic
    def expand(self, *sizes):
        return self.expand(*sizes)

    @storch.deterministic
    def expand_as(self, other):
        return self.expand_as(other)

    @storch.deterministic
    def fft(self, signal_ndim, normalized=False):
        return self.fft(signal_ndim, normalized)

    @storch.deterministic
    def flatten(self, start_dim=0, end_dim=-1):
        return self.flatten(start_dim, end_dim)

    @storch.deterministic
    def flip(self, dims):
        return self.flip(dims)

    @storch.deterministic
    def floor(self):
        return self.floor()

    @storch.deterministic
    def fmod(self, divisor):
        return self.fmod(divisor)

    @storch.deterministic
    def frac(self):
        return self.frac()

    @storch.deterministic
    def gather(self, dim, index):
        return self.gather(dim, index)

    @storch.deterministic
    def geqrf(self):
        return self.geqrf()

    @storch.deterministic
    def ger(self, vec2):
        return self.ger(vec2)

    def get_device(self):
        return self._tensor.get_device()

    @storch.deterministic
    def hardshrink(self, lambd=0.5):
        return self.hardshrink(lambd)

    @storch.deterministic
    def histc(self, bins=100, min=0, max=0):
        return self.histc(bins, min, max)

    @storch.deterministic
    def ifft(self, signal_ndim, normalized=False):
        return self.ifft(signal_ndim, normalized)

    @storch.deterministic
    def imag(self):
        return self.imag()

    @storch.deterministic
    def index_add(self, dim, index, tensor):
        return self.index_add(dim, index, tensor)

    @storch.deterministic
    def index_copy(self, dim, index, tensor):
        return self.index_copy(dim, index, tensor)

    @storch.deterministic
    def index_fill(self, dim, index, val):
        return self.index_fill(dim, index, val)

    @storch.deterministic
    def index_put(self, indices, value, accumulate=False):
        return self.index_add(indices, value, accumulate)

    @storch.deterministic
    def index_select(self, dim, index):
        return self.index_select(dim, index)

    @storch.deterministic
    def indices(self):
        return self.indices()

    @storch.deterministic
    def inverse(self):
        return self.inverse()

    @storch.deterministic
    def irfft(self, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
        return self.irfft(signal_ndim, normalized, onesided, signal_sizes)

    def is_pinned(self):
        return self._tensor.is_pinned()

    def is_shared(self):
        return self._tensor.is_shared()

    @storch.deterministic
    def kthvalue(self, k, dim=None, keepdim=False):
        if dim:
            return self.kthvalue(k, dim, keepdim)
        return self.kthvalue(k)

    @storch.deterministic
    def lerp(self):
        return self.lerp()

    @storch.deterministic
    def lgamma(self):
        return self.lgamma()

    @storch.deterministic
    def log(self):
        return self.log()

    @storch.deterministic
    def logdet(self):
        return self.logdet()

    @storch.deterministic
    def log10(self):
        return self.log10()

    @storch.deterministic
    def log1p(self):
        return self.log1p()

    @storch.deterministic
    def log2(self):
        return self.log2()

    @storch.deterministic
    def logsumexp(self, dim=None, keepdim=False):
        if dim:
            return self.logsumexp(dim, keepdim)
        return self.logsumexp()

    @storch.deterministic
    def lstsq(self, A):
        return self.lstsq(A)

    @storch.deterministic
    def lu(self, pivot=True, get_infos=False):
        return self.lu(pivot, get_infos)

    @storch.deterministic
    def lu_solve(self, LU_data, LU_pivots):
        return self.lu_solve(LU_data, LU_pivots)

    @storch.deterministic
    def masked_scatter(self, mask, tensor):
        return self.masked_scatter(mask, tensor)

    @storch.deterministic
    def masked_fill(self, mask, value):
        return self.masked_fill(mask, value)

    @storch.deterministic
    def masked_select(self, mask):
        return self.masked_select(mask)

    @storch.deterministic
    def matmul(self, other):
        return self.matmul(other)

    @storch.deterministic
    def matrix_power(self, n):
        return self.matrix_power(n)

    @storch.deterministic
    def max(self, dim=None, keepdim=False):
        if dim:
            return self.max(dim, keepdim)
        return self.max()

    @storch.deterministic
    def mean(self, dim=None, keepdim=False):
        if dim:
            return self.mean(dim, keepdim)
        return self.mean()

    @storch.deterministic
    def median(self, dim=None, keepdim=False):
        if dim:
            return self.median(dim, keepdim)
        return self.median()

    @storch.deterministic
    def min(self, dim=None, keepdim=False):
        if dim:
            return self.min(dim, keepdim)
        return self.min()

    @storch.deterministic
    def mm(self, other):
        return self.mm(other)

    @storch.deterministic
    def mode(self, dim=None, keepdim=False):
        if dim:
            return self.mode(dim, keepdim)
        return self.mode()

    @storch.deterministic
    def mv(self, other):
        return self.mv(other)

    @storch.deterministic
    def mvlgamma(self, p):
        return self.mvlgamma(p)

    @storch.deterministic
    def narrow(self, dimension, start, length):
        return self.narrow(dimension, start, length)

    @storch.deterministic
    def narrow_copy(self, dimension, start, length):
        return self.narrow_copy(dimension, start, length)

    @storch.deterministic
    def norm(self, p='fro', dim=None, keepdim=False, dtype=None):
        return self.norm(p, dim, keepdim, dtype)

    @storch.deterministic
    def nonzero(self):
        return self.nonzero()

    @storch.deterministic
    def orgqr(self, input2):
        return self.orgqr(input2)

    @storch.deterministic
    def ormqr(self, input2, input3, left=True, transpose=False):
        return self.ormqr(input2, input3, left, transpose)

    @storch.deterministic
    def permute(self, *dims):
        return self.permute(*dims)

    @storch.deterministic
    def pin_memory(self):
        return self.pin_memory()

    @storch.deterministic
    def pinverse(self):
        return self.pinverse()

    @storch.deterministic
    def polygamma(self, n):
        return self.polygamma(n)

    @storch.deterministic
    def pow(self, other):
        return self.pow(other)

    @storch.deterministic
    def prod(self, dim=None, keepdim=False):
        if dim:
            return self.prod(dim, keepdim)
        return self.prod()

    @storch.deterministic
    def qr(self, some=True):
        return self.qr(some)

    @storch.deterministic
    def q_per_channel_scales(self):
        return self.q_per_channel_scales()

    @storch.deterministic
    def q_per_channel_zero_points(self):
        return self.q_per_channel_zero_points()

    @storch.deterministic
    def reciprocal(self):
        return self.reciprocal()

    def register_hook(self, hook):
        self._tensor.register_hook(hook)

    @storch.deterministic
    def remainder(self, other):
        return self.remainder(other)

    @storch.deterministic
    def real(self):
        return self.real()

    @storch.deterministic
    def renorm(self, p, dim, maxnorm):
        return self.renorm(p, dim, maxnorm)

    @storch.deterministic
    def repeat(self, *sizes):
        return self.repeat(*sizes)

    def repeat_interleave(self, repeats, dim=None):
        return self.repeat_interleave(repeats, dim)

    @storch.deterministic
    def reshape(self, *shape):
        return self.reshape(*shape)

    @storch.deterministic
    def reshape_as(self, other):
        return self.reshape_as(other)

    def retain_grad(self):
        self._tensor.retain_grad()

    @storch.deterministic
    def roll(self, shifts, dim):
        return self.roll(shifts, dim)

    @storch.deterministic
    def rot90(self, k, dims):
        return self.rot90(k, dims)

    @storch.deterministic
    def round(self):
        return self.round()

    @storch.deterministic
    def rsqrt(self):
        return self.rsqrt()

    @storch.deterministic
    def scatter(self, dim, index, source):
        return self.scatter(dim, index, source)

    @storch.deterministic
    def scatter_add(self, dim, index, source):
        return self.scatter_add(dim, index, source)

    @storch.deterministic
    def select(self, dim, index):
        return self.select(dim, index)

    @storch.deterministic
    def sigmoid(self):
        return self.sigmoid()

    @storch.deterministic
    def sign(self):
        return self.sign()

    @storch.deterministic
    def sin(self):
        return self.sin()

    @storch.deterministic
    def sinh(self):
        return self.sinh()

    @storch.deterministic
    def slogdet(self):
        return self.slogdet()

    @storch.deterministic
    def solve(self, other):
        return self.solve(other)

    @storch.deterministic
    def sort(self, dim=None, descending=False):
        if dim:
            return self.mode(dim, descending)
        return self.sort(descending=descending)

    @storch.deterministic
    def sparse_mask(self, mask):
        return self.sparse_mask(mask)

    @storch.deterministic
    def sqrt(self):
        return self.sqrt()

    @storch.deterministic
    def squeeze(self, dim=None):
        if dim:
            return self.squeeze(dim)
        return self.squeeze()

    @storch.deterministic
    def std(self, dim=None, keepdim=False, unbiased=True):
        if dim:
            return self.std(dim, unbiased, keepdim)
        return self.std(unbiased=unbiased)

    @storch.deterministic
    def sum(self, dim=None, keepdim=False):
        if dim:
            return self.sum(dim, keepdim)
        return self.sum()

    @storch.deterministic
    def sum_to_size(self, *shape):
        return self.sum_to_size(*shape)

    @storch.deterministic
    def svd(self, some=True, compute_uv=True):
        return self.svd(some, compute_uv)

    @storch.deterministic
    def symeig(self, eigenvectors=False, upper=True):
        return self.symeig(eigenvectors, upper)

    @storch.deterministic
    def t(self):
        return self.t()

    @storch.deterministic
    def take(self, indices):
        return self.take(indices)

    @storch.deterministic
    def tan(self):
        return self.tan()

    @storch.deterministic
    def tanh(self):
        return self.tanh()

    @storch.deterministic
    def to(self, *args, **kwargs):
        return self.to(*args, **kwargs)

    @storch.deterministic
    def topk(self, k, dim=None, largest=True, sorted=True):
        if dim:
            return self.topk(k, dim, largest, sorted)
        return self.sum(k, largest=largest, sorted=sorted)

    @storch.deterministic
    def to_sparse(self, indices):
        return self.to_sparse(indices)

    @storch.deterministic
    def trace(self):
        return self.trace()

    @storch.deterministic
    def transpose(self, dim1, dim2):
        return self.transpose(dim1, dim2)

    @storch.deterministic
    def triangular_solve(self, A, upper=True, transpose=False, unitriangular=False):
        return self.triangular_solve(A, upper, transpose, unitriangular)

    @storch.deterministic
    def tril(self, k=0):
        return self.tril(k)

    @storch.deterministic
    def triu(self, k=0):
        return self.triu(k)

    @storch.deterministic
    def trunc(self):
        return self.trunc()

    @storch.deterministic
    def type(self, dtype=None, non_blocking=False, **kwargs):
        if dtype:
            return self.type(dtype, non_blocking, **kwargs)
        return self.type(non_blocking=False, **kwargs)

    @storch.deterministic
    def type_as(self, other):
        return self.type_as(other)

    @storch.deterministic
    def unbind(self, dim=0):
        return self.unbind(dim)

    @storch.deterministic
    def unfold(self, dimension, size, step):
        return self.unfold(dimension, size, step)

    @storch.deterministic
    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        if dim:
            return self.unique(sorted, return_inverse, return_counts, dim)
        return self.unique(sorted, return_inverse, return_counts)

    @storch.deterministic
    def unique_consecutive(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        if dim:
            return self.unique_consecutive(sorted, return_inverse, return_counts, dim)
        return self.unique_consecutive(sorted, return_inverse, return_counts)

    @storch.deterministic
    def unsqueeze(self, dim=None):
        if dim:
            return self.unsqueeze(dim)
        return self.unsqueeze()

    @storch.deterministic
    def values(self):
        return self.values()

    @storch.deterministic
    def var(self, dim=None, unbiased=True, keepdim=False):
        if dim:
            return self.var(dim, unbiased, keepdim)
        return self.var(unbiased=unbiased, keepdim=keepdim)

    @storch.deterministic
    def view(self, *shape):
        return self.view(*shape)

    @storch.deterministic
    def view_as(self, other):
        return self.view_as(other)

    #endregion

    #endregion

class DeterministicTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, parents, batch_links: [StochasticTensor], is_cost: bool):
        super().__init__(tensor, parents, batch_links)
        self._is_cost = is_cost
        if is_cost:
            storch.inference._cost_tensors.append(self)

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def is_cost(self) -> bool:
        return self._is_cost


class StochasticTensor(Tensor):
    def __init__(self, tensor: torch.Tensor, parents, sampling_method: storch.Method, batch_links: [StochasticTensor],
                 distribution: Distribution, requires_grad: bool, n: int):
        if n > 1:
            batch_links.insert(0, self)
        self.n = n
        self.distribution = distribution
        super().__init__(tensor, parents, batch_links)
        self.sampling_method = sampling_method
        self._requires_grad = requires_grad
        self._accum_grads = {}
        self._grad = None

    @property
    def stochastic(self):
        return True

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def grad(self):
        return self._accum_grads

    def total_expected_grad(self) -> Dict[str, torch.Tensor]:
        r = {}
        indices = self.batch_dim_indices()
        for tensor, grad in self._accum_grads.items():
            if grad.dim() == tensor.dim():
                r[tensor.param_name] = grad
            else:
                r[tensor.param_name] = grad.mean(dim=indices)
        return r

    def total_variance_grad(self) -> Dict[str, torch.Tensor]:
        """
        Computes the total variance on the gradient of the parameters of this distribution over all simulations .
        :return:
        """
        r = {}
        indices = self.batch_dim_indices()
        for tensor, grad in self._accum_grads.items():
            if grad.dim() == tensor.dim():
                raise ValueError("There are no batched dimensions to take statistics over. Make sure to call backwards "
                                 "with accum_grad=True")
            expected = grad.mean(dim=indices)
            diff = grad - expected
            squared_diff = diff * diff
            sse = squared_diff.sum(dim=indices)
            r[tensor.param_name] = sse.mean()
        return r



from storch.util import has_backwards_path