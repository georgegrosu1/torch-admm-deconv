import torch
import torch.nn as nn
from admmtor.elayers.channel_pool import ChannelPool
from admmtor.modelbuild.blocks import (
    MultiADMM
)


DECONV1 = {'kern_size': (),
         'max_iters': 100,
         'iso': True}
DECONV2 = {'kern_size': (),
         'max_iters': 100,
         'iso': True}


class ANet(nn.Module):
    def __init__(self,
                 in_nc: int = 3,
                 out_nc: int = 3,
                 nc: int = 64):
        super(ANet, self).__init__()

        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nc = nc
        self.admms_cfg = [DECONV1, DECONV2]
        self.blocks = nn.ModuleList()
        self.activation = nn.Sigmoid()
        
        head_in_nc = self.in_nc * len(self.admms_cfg) if self.admms_cfg is not None else self.in_nc
        self.admms_pool = ChannelPool(self.in_nc, soft=True, in_channels=head_in_nc)
        self._init_admms()
        self._init_conv_head()
        self._init_blocks()
        
    def _init_admms(self) -> None:
        self.admms = MultiADMM(
            self.admms_cfg
        ) if self.admms_cfg is not None else nn.Identity()
        
    def _init_conv_head(self) -> None:
        head_in_nc = self.in_nc * len(self.admms_cfg) if self.admms_cfg is not None else self.in_nc
        self.conv_head = nn.Conv2d(
            in_channels=head_in_nc,
            out_channels=self.nc,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    def _init_blocks(self) -> None:
        pass
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admms = self.admms(x)
        best_admms = self.admms_pool(admms)
        x1 = self.conv_head(admms)
        pass
