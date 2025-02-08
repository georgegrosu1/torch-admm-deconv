import torch
import torch.nn as nn


class ADMMWeightClipper(object):
    def __init__(self, keep_range: tuple[float, float]):
        self.keep_range = keep_range

    def __call__(self, module: nn.Module):
        # filter the variables to get the ones you want
        if hasattr(module, 'w'):
            module.w.data = self._clamp_vars(module.w.data)

    def _clamp_vars(self, wvars):
        wvars = torch.clamp(wvars, *self.keep_range)
        return wvars


class ADMMClipper(object):
    def __init__(self, max_val: float):
        self.keep_range = (1e-9, max_val)

    def __call__(self, module: nn.Module):
        if hasattr(module, 'lmbda'):
            module.lmbda.data = self._clamp_vars(module.lmbda.data)
        if hasattr(module, 'rho'):
            module.rho.data = self._clamp_vars(module.rho.data)
        if hasattr(module, 'bias'):
            module.bias.data = self._clamp_vars(module.rho.data)

    def _clamp_vars(self, wvars):
        wvars = torch.clamp(wvars, *self.keep_range)
        return wvars