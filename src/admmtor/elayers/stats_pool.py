from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class AbsLPPool2d(nn.Module):
    """LPPool2d wrapper that takes absolute values before pooling.

    This avoids NaNs that may arise from negative intermediate sums when the
    implementation computes x.pow(p).sum() followed by pow(sum, 1.0/p).
    Using `abs` guarantees the inner quantity is non-negative and prevents
    numerical NaNs for odd `p` values.
    """

    def __init__(self, norm_type: int = 2, kernel_size=1, stride=None, ceil_mode: bool = False, cons: float = 1e-12):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode
        self.cons = cons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.lp_pool2d(x.abs() + self.cons, self.norm_type, self.kernel_size, self.stride, self.ceil_mode)
    
    
class GeoMeanPool2d(nn.Module):
    """Geometric mean pooling layer that computes the geometric mean of each patch."""

    def __init__(self, kernel_size=1, stride=None, cons: float = 1e-12):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.cons = cons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_x = torch.log(x.abs() + self.cons)
        pooled_log = F.avg_pool2d(log_x, self.kernel_size, self.stride)
        return torch.exp(pooled_log)
    
    
class AdaptiveGeoMeanPool2d(nn.Module):
    """Adaptive geometric mean pooling layer that computes the geometric mean of each patch."""

    def __init__(self, output_size: int | tuple[int, int], cons: float = 1e-12):
        super().__init__()
        self.output_size = output_size
        self.cons = cons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_x = torch.log(x.abs() + self.cons)
        pooled_log = F.adaptive_avg_pool2d(log_x, self.output_size)
        return torch.exp(pooled_log)
