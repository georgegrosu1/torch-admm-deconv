import torch
import torch.nn as nn
from admmtor.elayers.channel_pool import ChannelPool
from admmtor.modelbuild.blocks import (
    default_init_weights,
    MultiADMM,
    CoarseBlock,
    FineBlock,
    FusionBlock
)


class ANetBlock(nn.Module):
    def __init__(self,
                 in_nc: int = 3,
                 out_nc: int = 3,
                 nc: int = 64,
                 c_mul: int = 2,
                 activation: nn.Module = nn.Identity()):
        super(ANetBlock, self).__init__()
        
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nc = nc
        self.c_mul = c_mul
        self.activation = activation
        self.coarse_layer = CoarseBlock(self.nc, self.c_mul)
        self.fine_layer = FineBlock(self.nc, self.c_mul)
        self.fusion_layer = FusionBlock(self.in_nc, self.nc, self.out_nc, self.c_mul)
        self.top_ch_pool = ChannelPool(top_k=self.out_nc, temperature=0.8, soft=True, in_channels=self.nc)
        
    def forward(self, x: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        top_x_ch = self.top_ch_pool(x)
        fine_out = self.fine_layer(x)
        coarse_out = self.coarse_layer(x)
        fused_out = self.fusion_layer(img, coarse_out, fine_out)
        return self.activation(top_x_ch + fused_out)


class ANet(nn.Module):
    def __init__(self,
                 in_nc: int = 3,
                 out_nc: int = 3,
                 nc: int = 64,
                 num_fusion_blocks: int = 4,
                 c_mul: int = 4,
                 admms_cfg: list[dict] = None):
        super(ANet, self).__init__()

        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nc = nc
        self.num_fusion_blocks = num_fusion_blocks
        self.admms_cfg = admms_cfg
        self.activation = nn.Sigmoid()
        self.c_mul = c_mul

        head_in_nc = self.in_nc * len(self.admms_cfg) if self.admms_cfg is not None else self.in_nc
        self.admms_pool = ChannelPool(top_k=self.in_nc, 
                                      soft=True,
                                      normalize_weights=True, 
                                      differentiable=True, 
                                      in_channels=head_in_nc)
        self.intermediate_pool = ChannelPool(top_k=self.out_nc, 
                                             temperature=0.8,
                                             normalize_weights=True, 
                                             differentiable=True,
                                             soft=True, 
                                             in_channels=self.nc)
        self.multiadmm = self._init_multiadmm()
        self.conv_head = self._init_conv_head()
        self.anet_blocks = self._init_anet_blocks()
        self.final_block = ANetBlock(self.in_nc, self.out_nc, self.nc, self.c_mul, self.activation)
        # self.apply(default_init_weights)
        
        
    def _init_multiadmm(self) -> nn.Module:
        return MultiADMM(
            self.admms_cfg
        ) if self.admms_cfg is not None else nn.Identity()
        
    def _init_conv_head(self) -> nn.Module:
        head_in_nc = self.in_nc * (len(self.admms_cfg) + 1) if self.admms_cfg is not None else self.in_nc
        return nn.Conv2d(
            in_channels=head_in_nc,
            out_channels=self.nc,
            kernel_size=3,
            stride=1, 
            dilation=2,
            padding='same',
            padding_mode='circular')
        
    def _init_anet_blocks(self) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for _ in range(self.num_fusion_blocks):
            blocks.append(
                ANetBlock(
                    in_nc=self.in_nc,
                    out_nc=self.nc,
                    nc=self.nc,
                    c_mul=self.c_mul,
                )
            )
        return blocks
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        admms = self.multiadmm(x)
        best_admm = self.admms_pool(admms)
        out = self.conv_head(torch.cat([x, admms], dim=1))
        for block in self.anet_blocks:
            out = block(out, best_admm)
            best_admm += self.intermediate_pool(out)
        return self.activation(best_admm + self.final_block(out, best_admm))
