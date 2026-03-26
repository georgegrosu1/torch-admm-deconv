from __future__ import annotations

import torch
from torch import nn
from admmtor.elayers.stats_pool import AdaptiveGeoMeanPool2d


class GeometricGating(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adaptgeomean = AdaptiveGeoMeanPool2d(output_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1_geomean = self.adaptgeomean(x1)
        x2_geomean = self.adaptgeomean(x2)
        return torch.cat([x1 * x2_geomean, x2 * x1_geomean], dim=1)