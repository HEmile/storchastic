# from typing import Callable, Union
# from torch import Tensor
# from storch import StochasticTensor, stochastic
# # BlackboxTensor
# from storch.typing import AnyBlackboxTensor
#
# class MDP:
#
#     def __init__(self, transition_func: Callable[[Tensor, Tensor], (Tensor, Tensor)], action_func: Callable[[Tensor], StochasticTensor], discount):
#         self.transition_func = stochastic(transition_func)
#         self.action_func = stochastic(action_func)
#         self.discount = discount
#
#     def simulate(self, state: AnyBlackboxTensor, steps=1):
#         for i in range(steps):
#             action_tensor = self.action_func(state)
#             if isinstance(state, BlackboxTensor)
#             if len(action_tensor.batch_links) > 0:
#                 for indices in action_tensor.iterate_batch_indices():
#
