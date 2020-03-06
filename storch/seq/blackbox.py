from storch import Tensor

class BlackboxTensor(Tensor):
    def __init__(self, tensor, parents, batch_links):
        super().__init__(tensor, parents, batch_links)
        self.tensor = tensor
        self.parents = parents
        self.batch_links = batch_links