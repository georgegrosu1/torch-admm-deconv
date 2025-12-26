import torch
import torch.nn as nn
from admmtor.modelbuild.blocks import (
    MultiADMM
)


class DenoiserV2(nn.Module):
    def __init__(self):
        super(DenoiserV2, self).__init__()

        self.blocks = nn.ModuleList()
        self.activation = nn.Sigmoid()
        
        self._init_blocks()
        
    def _init_blocks(self) -> None:
        pass
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
