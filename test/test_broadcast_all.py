import torch


class TensorLike:
    def __torch_function__(func, types, args=(), kwargs=None):
        print("In __torch_function_")


tensor = torch.tensor([1.2, 3.4, 5.6])
tensor_like = TensorLike()
# tensor.expand_as(TensorLike())
from torch.distributions import Normal

# Normal(loc=TensorLike(), scale=1)
tensor_like + tensor
