from typing import Optional, List

from storch.tensor import Tensor, StochasticTensor, CostTensor, IndependentTensor
import torch
from storch.util import print_graph
import storch


_cost_tensors: [CostTensor] = []
_sampling_methods: [storch.Method] = []


def denote_independent(
    tensor: torch.Tensor,
    dim: int,
    plate_name: str,
    weight: Optional[storch.Tensor] = None,
) -> IndependentTensor:
    """
    Denote the given dimensions on the tensor as being independent, that is, batched dimensions.
    It will automatically put these dimensions to the left.
    :param tensor:
    :param dims:
    :param plate_name: Name of the plate. Reused if called again
    :return:
    """
    if (
        storch.wrappers._context_stochastic
        or storch.wrappers._context_deterministic > 0
    ):
        raise RuntimeError(
            "Cannot create independent tensors within a deterministic or stochastic context."
        )
    if isinstance(tensor, storch.Tensor):
        t_tensor = tensor._tensor
        if dim != 0:
            t_tensor = t_tensor.transpose(dim, 0)
        tensor_name = (
            tensor.name + "_indep_" + plate_name if tensor.name else plate_name
        )
        return IndependentTensor(
            t_tensor, [tensor], tensor.plates, tensor_name, plate_name, weight
        )
    else:
        if dim != 0:
            tensor = tensor.transpose(dim, 0)
        return IndependentTensor(tensor, [], [], plate_name, plate_name, weight)


def gather_samples(
    samples: List[storch.Tensor],
    plate_name: str,
    weight: Optional[storch.Tensor] = None,
) -> IndependentTensor:
    if (
        storch.wrappers._context_stochastic
        or storch.wrappers._context_deterministic > 0
    ):
        raise RuntimeError(
            "Cannot create independent tensors within a deterministic or stochastic context."
        )
    collect_tensors = []
    for sample in samples:
        sample = sample._tensor
        if not sample.shape[0] == 1:
            sample = sample.unsqueeze(0)
        collect_tensors.append(sample)
    cat_tensors = torch.cat(collect_tensors, 0)
    tensor_name = (
        samples[0].name + "_indep_" + plate_name if samples[0].name else plate_name
    )
    return IndependentTensor(
        cat_tensors, samples, samples[0].plates.copy(), tensor_name, plate_name, weight
    )


def add_cost(cost: Tensor, name: str):
    if cost.event_shape != ():
        if cost.event_shape == (1,):
            cost = cost.squeeze(-1)
        else:
            raise ValueError("Can only register cost functions with empty event shapes")
    if not name:
        raise ValueError(
            "No name provided to register cost node. Make sure to register an unique name with the cost."
        )
    cost = CostTensor(cost._tensor, [cost], cost.plates, name)
    if torch.is_grad_enabled():
        storch.inference._cost_tensors.append(cost)
    return cost


def backward(retain_graph=False, debug=False, print_costs=False) -> torch.Tensor:
    """

    :param retain_graph: If set to False, it will deregister the added cost nodes. Should usually be set to False.
    :param debug: Prints debug information on the backwards call.
    :param accum_grads: Saves gradient information in stochastic nodes. Note that this is an expensive option as it
    requires doing O(n) backward calls for each stochastic node sampled multiple times. Especially if this is a
    hierarchy of multiple samples.
    :return:
    """
    costs: [storch.Tensor] = storch.inference._cost_tensors
    if debug:
        print_graph(costs)

    # Sum of averages of cost node tensors
    total_cost = 0.0
    # Sum of losses that can be backpropagated through in keepgrads without difficult iterations
    accum_loss = 0.0

    stochastic_nodes = set()
    # Loop over different cost nodes
    for c in costs:
        # Do not detach the weights when reducing. This is used in for example expectations to weight the
        # different costs.
        reduced_cost = storch.reduce_plates(c, detach_weights=False)

        if print_costs:
            print(c.name, ":", reduced_cost._tensor.item())
        total_cost += reduced_cost
        # Compute gradients for the cost nodes themselves, if they require one.
        if reduced_cost.requires_grad:
            accum_loss += reduced_cost
        for parent in c.walk_parents(depth_first=False):
            # Instance check here instead of parent.stochastic, as backward methods are only used on these.
            if isinstance(parent, StochasticTensor):
                stochastic_nodes.add(parent)
            else:
                continue
            if (
                not parent.requires_grad
                or not parent.sampling_method
                or not parent.sampling_method.adds_loss(parent, c)
            ):
                continue

            # Transpose the parent stochastic tensor, so that its shape is the same as the cost but the event shape, and
            # possibly extra dimensions...?
            parent_tensor = parent._tensor
            reduced_cost = c
            parent_plates = parent.multi_dim_plates()

            # Reduce all plates that are in the cost node but not in the parent node
            for plate in storch.order_plates(c.multi_dim_plates(), reverse=True):
                if plate not in parent_plates:
                    reduced_cost = plate.reduce(reduced_cost, detach_weights=True)

            # Align the parent tensor so that the plate dimensions are in the same order as the cost tensor
            for index_c, plate in enumerate(reduced_cost.multi_dim_plates()):
                index_p = parent_plates.index(plate)
                if index_c != index_p:
                    parent_tensor = parent_tensor.transpose(index_p, index_c)
                    parent_plates[index_p], parent_plates[index_c] = (
                        parent_plates[index_c],
                        parent_plates[index_p],
                    )
            # Add empty (k=1) plates to new parent
            for plate in parent.plates:
                if plate not in parent_plates:
                    parent_plates.append(plate)

            # Create new storch Tensors with different order of plates for the cost and parent
            new_parent = storch.tensor.StochasticTensor(
                parent_tensor,
                [],
                parent_plates,
                parent.name,
                parent.n,
                parent.sampling_method,
                parent.distribution,
                parent._requires_grad,
            )
            # Fake the new parent to be the old parent within the graph by mimicing its place in the graph
            new_parent._parents = parent._parents
            for p, has_link in new_parent._parents:
                p._children.append((new_parent, has_link))
            new_parent._children = parent._children
            cost_per_sample = parent.sampling_method._estimator(
                new_parent, reduced_cost
            )

            if cost_per_sample is not None:
                # The backwards call for reparameterization happens in the
                # backwards call for the costs themselves.
                # Now mean_cost has the same shape as parent.batch_shape
                final_reduced_cost = storch.reduce_plates(
                    cost_per_sample, detach_weights=True
                )
                if final_reduced_cost.ndim == 1:
                    final_reduced_cost = final_reduced_cost.squeeze(0)
                accum_loss += final_reduced_cost

    if isinstance(accum_loss, storch.Tensor) and accum_loss._tensor.requires_grad:
        accum_loss._tensor.backward(retain_graph=retain_graph)

    for s_node in stochastic_nodes:
        if s_node.sampling_method:
            s_node.sampling_method._update_parameters()

    if not retain_graph:
        reset()

    # TODO: How much does accum_loss really say? Should we really keep it? We want to minimize total_cost, anyways.
    return total_cost._tensor  # , accum_loss._tensor


def reset():
    storch.inference._cost_tensors = []
    for method in storch.inference._sampling_methods:
        method.reset()
    storch.inference._sampling_methods = []
