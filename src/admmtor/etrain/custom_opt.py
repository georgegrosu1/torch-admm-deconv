import torch


class MultiOptimizers(torch.optim.Optimizer):
    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def step(self, closure=None):
        for optimizer in self.optimizers:
            optimizer.step(closure)

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
