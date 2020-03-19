from storch import Tensor


class BlackboxTensor(Tensor):
    def __init__(self, tensor, parents, plates):
        super().__init__(tensor, parents, plates)
        self.tensor = tensor
        self.parents = parents
        self.batch_links = plates
