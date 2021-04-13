from typing import Optional, List, Union

from storch.tensor import Tensor, StochasticTensor, CostTensor, IndependentTensor
import torch
from storch.util import print_graph
import storch


_cost_tensors: [CostTensor] = []
_sampling_methods: [storch.method.Method] = []


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
    samples: Union[List[storch.Tensor], List[torch.Tensor]],
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
        if isinstance(sample, storch.Tensor):
            sample = sample._tensor
        sample = sample.unsqueeze(0)
        collect_tensors.append(sample)
    cat_tensors = storch.cat(collect_tensors, 0)

    if isinstance(samples[0], storch.Tensor):
        tensor_name = (
            samples[0].name + "_indep_" + plate_name if samples[0].name else plate_name
        )
        return IndependentTensor(
            cat_tensors,
            samples,
            samples[0].plates.copy(),
            tensor_name,
            plate_name,
            weight,
        )
    return IndependentTensor(
        cat_tensors, [], [], "_indep_" + plate_name, plate_name, weight
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


def magic_box(l: storch.Tensor):
    """
    Implements the MagicBox operator from
    DiCE: The Infinitely Differentiable Monte-Carlo Estimator https://arxiv.org/abs/1802.05098
    It returns 1 in the forward pass, but returns magic_box(l) \cdot r in the backwards pass.
    This allows for any-order gradient estimation.
    """
    return (l - l.detach()).exp()


def backward(
    debug: bool = False,
    create_graph: bool = False,
    update_estimator_params: bool = True,
) -> torch.Tensor:
    """
    Computes the gradients of the cost nodes with respect to the parameter nodes. It uses the storch
    methods used to sample stochastic nodes to properly estimate their gradient.

    Args:
        debug: Prints debug information on the backwards call.
        accum_grads: Saves gradient information in stochastic nodes. Note that this is an expensive option as it
        requires doing O(n) backward calls for each stochastic node sampled multiple times. Especially if this is a
        hierarchy of multiple samples.
        create_graph (bool): Creates the backpropagation graph of the gradient estimation for higher-order derivatives
    Returns:
        torch.Tensor: The average total cost normalized by the sampling weights.
    """

    costs: [storch.Tensor] = storch.inference._cost_tensors
    if not costs:
        raise RuntimeError("No cost nodes registered for backward call.")
    if debug:
        print_graph(costs)

    # Sum of averages of cost node tensors
    total_cost = 0.0

    stochastic_nodes = set()
    _create_graph = create_graph
    # Loop over different cost nodes
    for c in costs:
        # Do not detach the weights when reducing. This is used in for example expectations to weight the
        # different costs.
        # reduced_cost = storch.reduce_plates(c, detach_weights=False)
        #
        # if print_costs:
        #     print(c.name, ":", reduced_cost._tensor.item())
        # total_cost += reduced_cost
        # Compute gradients for the cost nodes themselves, if they require one.
        # if reduced_cost.requires_grad:
        #     accum_loss += reduced_cost

        L = c._tensor.new_tensor(0.0)
        cost_loss = 0.0
        for parent in c.walk_parents(depth_first=False):
            # Instance check here instead of parent.stochastic, as backward methods are only used on these.
            if isinstance(parent, StochasticTensor):
                stochastic_nodes.add(parent)
            else:
                continue
            if not parent.requires_grad or not parent.method:
                continue
            _create_graph = (
                parent.method.should_create_higher_order_graph() or _create_graph
            )
            if not parent.method.adds_loss(parent, c):
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
                parent.distribution,
                parent._requires_grad,
                parent.method,
            )
            new_parent.param_grads = parent.param_grads
            # Fake the new parent to be the old parent within the graph by mimicking its place in the graph
            new_parent._parents = parent._parents
            for p, has_link in new_parent._parents:
                p._children.append((new_parent, has_link))
            new_parent._children = parent._children

            # Compute the estimator triple
            (
                # TODO: baseline shouldn't be a separate thing
                gradient_function,
                baseline,
                control_variate,
            ) = parent.method._estimator(new_parent, reduced_cost)

            _A = 0.0
            if gradient_function is not None:
                L = L + gradient_function
                if baseline is not None:
                    _A = baseline.detach() * (
                        1 - magic_box(gradient_function)
                    )

            if control_variate is not None:
                _A += control_variate - control_variate.detach()

            if baseline is not None or control_variate is not None:
                final_A = storch.reduce_plates(
                    _A, detach_weights=True
                )
                if final_A.ndim == 1:
                    final_A = final_A.squeeze(0)
                cost_loss += final_A
        cost_loss += storch.reduce_plates(magic_box(L) * c, detach_weights=False)
        total_cost += cost_loss

    if isinstance(total_cost, storch.Tensor) and total_cost._tensor.requires_grad:
        total_cost._tensor.backward(create_graph=_create_graph)

    if update_estimator_params:
        for s_node in stochastic_nodes:
            if s_node.method:
                s_node.method._update_parameters()

    if not create_graph:
        total_cost._clean()
        total_cost.grad_fn = None
        reset()

    return total_cost._tensor


def reset():
    # Free the SC graph links. This often improves garbage collection for larger graphs.
    # Unfortunately Python's GC seems to have imperfect cycle detection?
    for c in storch.inference._cost_tensors:
        c._clean()

    storch.inference._cost_tensors = []
    for method in storch.inference._sampling_methods:
        method.reset()
    storch.inference._sampling_methods = []
