from torch.distributions import Distribution
from storch.tensor import Tensor, StochasticTensor, DeterministicTensor
from storch.method import Method, Infer, ScoreFunction
import torch
from storch.util import print_graph, get_distr_parameters
import storch
from operator import mul
from itertools import product
from functools import reduce
from typing import Dict, Optional

_cost_tensors: [DeterministicTensor] = []
_backward_indices: Dict[StochasticTensor, int] = {}
_backward_cost: Optional[DeterministicTensor] = None
_keep_grad: bool = False

def _create_hook(sample: StochasticTensor, tensor: torch.tensor):
    event_shape = list(tensor.shape)
    def hook(grad:torch.Tensor):
        if not _keep_grad:
            return
        accum_grads = sample._accum_grads
        if tensor not in accum_grads:
            add_n = [sample.n] if sample.n > 1 else []
            accum_grads[tensor] = grad.new_zeros(add_n + event_shape)
        indices = []
        for link in sample.batch_links:
            indices.append(storch.inference._backward_indices[link])
        indices = tuple(indices)
        offset_indices = 1 if sample.n > 1 else 0
        accum_grads[tensor][indices] += grad[indices[offset_indices:]]
    return hook


def sample(distr: Distribution, method: Method = None, n: int = 1) -> Tensor:
    if not method:
        if distr.has_rsample:
            method = Infer()
        else:
            method = ScoreFunction()
    params = get_distr_parameters(distr, filter_requires_grad=True)

    tensor = method.sample(distr, n)
    if n == 1:
        tensor = tensor.squeeze(0)
    plates = storch.wrappers._plate_links.copy()
    s_tensor = StochasticTensor(tensor, storch.wrappers._stochastic_parents, method, plates, distr, len(params) > 0, n)
    for param in params:
        # TODO: Possibly could find the wrong gradients here if multiple distributions use the same parameter?
        param.register_hook(_create_hook(s_tensor, param))
    return s_tensor

def _keep_grads_backwards(surrounding_node: Tensor, backwards_tensor: torch.Tensor) -> torch.Tensor:
    normalize_factor = 1. / reduce(mul, surrounding_node.batch_shape)
    ranges = list(map(lambda a: list(range(a)), surrounding_node.batch_shape))
    total_loss = 0.
    for indices in product(*ranges):
        # Minimize each pass individually to be able to save gradient statistics over multiple runs
        loss = backwards_tensor[indices] * normalize_factor
        zipped_indices = {}
        for index_batch, index_value in enumerate(indices):
            zipped_indices[surrounding_node.batch_links[index_batch]] = index_value
        storch.inference._backward_indices = zipped_indices
        loss.backward(retain_graph=True)
        total_loss += loss
    return total_loss


def backward(retain_graph=False, debug=False, keep_grads=False):
    """

    :param retain_graph: If set to False, it will deregister the added cost nodes. Should usually be set to False.
    :param debug: Prints debug information on the backwards call.
    :param keep_grads: Saves gradient information in stochastic nodes. Note that this is an expensive option as it
    requires doing O(n) backward calls for each stochastic node sampled multiple times. Especially if this is a
    hierarchy of multiple samples.
    :return:
    """
    costs = storch.inference._cost_tensors
    storch.inference._keep_grad = keep_grads
    if debug:
        print_graph(costs)

    # Sum of averages of cost node tensors
    total_cost = 0.
    # Sum of losses that can be backpropagated through in keepgrads without difficult iterations
    accum_loss = 0.
    # Sum of all losses
    total_loss = 0.

    for c in costs:
        avg_cost = c._tensor.mean()
        total_cost += avg_cost
        storch.inference._backward_cost = c
        for parent in c.walk_parents(depth_first=False):
            if not parent.stochastic or not parent.requires_grad:
                continue

            # Sum out over the plate dimensions of the parent, so that the shape is the same as the parent but the event shape
            mean_cost = c._tensor
            c_indices = c.batch_links.copy()
            for index_p, plate in enumerate(parent.batch_links):
                index_c = c_indices.index(plate)
                if not index_c == index_p:
                    mean_cost = mean_cost.transpose(index_p, index_c)
                    c_indices[index_p], c_indices[index_c] = c_indices[index_c], c_indices[index_p]

            # Then take the mean over the resulting dimensions (ie, plates that are created by other samples)
            if len(parent.batch_links) != len(mean_cost.shape):
                sum_out_dims = tuple(range(len(parent.batch_links), len(mean_cost.shape)))
                mean_cost = mean_cost.mean(sum_out_dims)

            additive_terms = parent.sampling_method.estimator(parent, c, mean_cost)
            # This can be None for eg reparameterization. The backwards call for reparameterization happens in the
            # backwards call for the costs themselves.
            if additive_terms is not None:
                # Now mean_cost has the same shape as parent.batch_shape
                if keep_grads and len(parent.batch_shape) > 0:
                    total_loss += _keep_grads_backwards(parent, additive_terms)
                else:
                    at_mean = additive_terms.mean()
                    accum_loss += at_mean
                    total_loss += at_mean

        # Compute gradients for the cost nodes themselves
        if keep_grads and len(c.batch_shape) > 0:
            total_loss += _keep_grads_backwards(c, c._tensor)
        else:
            accum_loss += avg_cost
            total_loss += avg_cost

    if isinstance(accum_loss, torch.Tensor) and accum_loss.requires_grad:
        accum_loss.backward()

    if not retain_graph:
        storch.inference._cost_tensors = []

    return total_cost, total_loss
