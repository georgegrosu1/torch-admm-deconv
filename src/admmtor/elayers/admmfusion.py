import torch
import torch.nn as nn

from admmtor.elayers.admmdeconv import ADMMDeconv
from admmtor.elayers.cwa import ChannelCompression
from admmtor.elayers.attentionpool import AttentionChannelPooling


class ADMMFusion(nn.Module):
    def __init__(self,
                 admms_cfgs: list[dict],
                 in_channels: int,
                 compressions: list[ChannelCompression] = (ChannelCompression.STD, ChannelCompression.MEDIAN,
                                                           ChannelCompression.MAX, ChannelCompression.MEAN),
                 probas_channels_factor: int = 2,
                 reduce_probas_space: bool = False,
                 with_admms: bool = False):
        super(ADMMFusion, self).__init__()

        self.in_channels = in_channels
        self.admms_cfgs = admms_cfgs
        self.compressions = compressions
        self.with_admms = with_admms
        self.probas_channels_factor = probas_channels_factor
        self.reduce_probas_space = reduce_probas_space

        self.fusioned_channels_size = in_channels * len(admms_cfgs)
        self.admms = nn.ModuleList()
        for admmcfg in admms_cfgs:
            self.admms.append(ADMMDeconv(**admmcfg))
        self.acp = AttentionChannelPooling(self.fusioned_channels_size, in_channels, compressions,
                                           probas_channels_factor, reduce_probas_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([admm(x) for admm in self.admms], dim=1)
        if self.with_admms:
            return torch.cat([self.acp(x), x], dim=1)
        return self.acp(x)