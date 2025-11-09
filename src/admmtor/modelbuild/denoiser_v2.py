import torch
import torch.nn as nn
from admmtor.modelbuild.blocks import (
    DivergentAttention,
    LayerNorm2d,
    MultiScaleConvPool,
    MultiADMM
)
from admmtor.elayers.cwa import ChannelWiseAttention
from admmtor.elayers.attentionpool import AttentionChannelPooling


class RestorerV2Block(nn.Module):
    def __init__(self,
                 in_c: int,
                 filters: int,
                 out_c: int,
                 ks: list[int],
                 pads: list[int],
                 admms_dicts: list[dict] | None = None):
        super(RestorerV2Block, self).__init__()

        self.admms = MultiADMM(admms_dicts) if admms_dicts else None
        self.norm = LayerNorm2d(in_c, eps=1e-9)
        self.msconv1 = MultiScaleConvPool(in_c, filters, ks, pads, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RestorerV2(nn.Module):
    def __init__(self,
                 in_channels: int,
                 blocks_filters: list[int],
                 blocks_gate_channels: list[int],
                 blocks_attention_reduction: list[int],
                 admms: list[dict] = None):
        super(RestorerV2, self).__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass