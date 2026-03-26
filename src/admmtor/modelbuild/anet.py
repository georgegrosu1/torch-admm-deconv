import torch
import torch.nn as nn
from admmtor.elayers.channel_pool import ChannelPool
from admmtor.modelbuild.blocks import (
    MultiADMM
)


class ANet(nn.Module):
    def __init__(self,
                 in_nc: int = 3,
                 out_nc: int = 3,
                 nc: int = 64,
                 admms_cfg: list[dict] = None):
        super(ANet, self).__init__()

        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nc = nc
        self.admms_cfg = admms_cfg
        self.blocks = nn.ModuleList()
        self.activation = nn.Sigmoid()
        
        head_in_nc = self.in_nc * len(self.admms_cfg) if self.admms_cfg is not None else self.in_nc
        self.admms_pool = ChannelPool(self.in_nc, soft=True, in_channels=head_in_nc)
        self.multiadmm = self._init_multiadmm()
        self.conv_head = self._init_conv_head()
        self.refinement_blocks = self._init_refinement_blocks()
        
        self.fusion_blocks = self._init_fusion_blocks()
        
    def _init_multiadmm(self) -> nn.Module:
        return MultiADMM(
            self.admms_cfg
        ) if self.admms_cfg is not None else nn.Identity()
        
    def _init_conv_head(self) -> nn.Module:
        head_in_nc = self.in_nc * len(self.admms_cfg) if self.admms_cfg is not None else self.in_nc
        return nn.Conv2d(
            in_channels=head_in_nc,
            out_channels=self.nc,
            kernel_size=3,
            stride=1, 
            dilation=2,
            padding='same',
            padding_mode='circular',
            bias=True
        )
        
    def _init_refinement_blocks(self) -> nn.Module:
        pass
        
    def _init_fusion_blocks(self) -> nn.Module:
        pass
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admms = self.multiadmm(x)
        best_admm = self.admms_pool(admms)
        x1 = self.conv_head(admms)
        pass
