import torch


class TensorLike:
    def __torch_function__(func, types, args=(), kwargs=None):
        print("In __torch_function_")


from torch.distributions import Normal

Normal(loc=TensorLike(), scale=1)
