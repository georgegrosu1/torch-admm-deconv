import torch
import torch.nn as nn
from typing import List, Dict

from admmtor.elayers.admmdeconv import ADMMDeconv


class Deconvs(nn.Module):
    def __init__(self,
                 admms_args: List[Dict]):
        super(Deconvs, self).__init__()
        self._init_deconvs(admms_args)


    def _init_deconvs(self,
                        admms_args: List[Dict]):
        self.blocks = nn.ModuleList()
        for i in range(len(admms_args)):
            self.blocks.append(ADMMDeconv(**(admms_args[i])))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([block(x) for block in self.blocks], dim=1)
