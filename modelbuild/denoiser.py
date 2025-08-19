import torch
import torch.nn as nn
from modelbuild.blocks import DivergentAttention
from elayers.cwa import ChannelWiseAttention
from elayers.attentionpool import AttentionChannelPooling


class DivergentRestorer(nn.Module):
    def __init__(self,
                 level_branches: list,
                 in_channels: int,
                 final_channels: int,
                 filters: int,
                 gate_channels: int,
                 attention_reduction: int,
                 intermediate_activation: nn.Module = None,
                 output_activation: nn.Module = None,
                 admms: list[dict] = None):
        super(DivergentRestorer, self).__init__()

        num_levels = len(level_branches)
        self._level_branches = level_branches

        self.blocks = nn.ModuleList()
        self.scas = nn.ModuleList()
        for i in range(num_levels):
            self.scas.append(ChannelWiseAttention(filters))
            if i == 0:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=in_channels,
                                                      out_channels=filters,
                                                      conv_filters=filters,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=intermediate_activation,
                                                      admms=admms))
            elif i == num_levels - 1:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=filters + in_channels,
                                                      out_channels=final_channels,
                                                      conv_filters=filters,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=output_activation))
            else:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=filters + in_channels,
                                                      out_channels=filters,
                                                      conv_filters=filters,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=intermediate_activation))
                # self.top_ch.append(AttentionChannelPooling(filters, gate_channels * 2, attention_reduction,
                #                                       conv_filters=gate_channels * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        out = self.scas[0](out)
        for i in range(1, len(self.blocks)):
            if i < len(self.blocks) - 1:
                out = self.blocks[i](torch.cat(tensors=[out, x], dim=1))
                out = self.scas[i](out)
            else:
                out = self.scas[i](out)
                out = self.blocks[i](torch.cat(tensors=[out, x], dim=1))
        return out