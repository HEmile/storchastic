from abc import ABC, abstractmethod
import torch
from storch.tensor import StochasticTensor, DeterministicTensor

class Baseline(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_baseline(self, tensor: StochasticTensor, cost_node: DeterministicTensor,
                         costs: torch.Tensor) -> torch.Tensor:
        pass


class MovingAverageBaseline(Baseline):
    def __init__(self, exponential_decay=0.95):
        super().__init__()
        self.register_buffer("exponential_decay", torch.tensor(exponential_decay))
        self.register_buffer("moving_average", torch.tensor(0.))

    def compute_baseline(self, tensor: StochasticTensor, cost_node: DeterministicTensor, costs: torch.Tensor) -> torch.Tensor:
        avg_cost = costs.mean().detach()
        self.moving_average = self.exponential_decay * self.moving_average + (1 - self.exponential_decay) * avg_cost
        return self.moving_average


class BatchAverageBaseline(Baseline):
    def compute_baseline(self, tensor: StochasticTensor, cost_node: DeterministicTensor,
                         costs: torch.Tensor) -> torch.Tensor:
        if tensor.n == 1:
            raise ValueError("Can only use the batch average baseline if multiple samples are used. With n=1, you would"
                             "optimize the constant 0.")
        return costs.mean().detach()
