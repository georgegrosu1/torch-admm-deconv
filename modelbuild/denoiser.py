import torch
import torch.nn as nn
from modelbuild.blocks import DivergentAttention, TopNChannelPooling


class DivergentRestorer(nn.Module):
    def __init__(self,
                 num_levels: int,
                 init_branches: int,
                 in_channels: int,
                 final_channels: int,
                 divergence_factor: int,
                 filters: int,
                 gate_channels: int,
                 attention_reduction: int,
                 intermediate_activation: nn.Module = None,
                 output_activation: nn.Module = None,
                 admms: list[dict] = None):
        super(DivergentRestorer, self).__init__()

        self._level_branches = [init_branches]
        for i in range(1, num_levels):
            self._level_branches.append(self._level_branches[-1] * divergence_factor)

        self.blocks = nn.ModuleList()
        self.top_ch = nn.ModuleList()
        for i in range(num_levels):
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
                                                      in_channels=filters + 3,
                                                      out_channels=final_channels,
                                                      conv_filters=filters,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=output_activation))
            else:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=filters + 3,
                                                      out_channels=filters,
                                                      conv_filters=filters,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=intermediate_activation))
                self.top_ch.append(TopNChannelPooling(filters, gate_channels * 2, attention_reduction,
                                                      conv_filters=gate_channels * 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            if i < len(self.blocks) - 1:
                skip = out
                out = self.blocks[i](torch.cat(tensors=[out, x], dim=1))
                out = self.top_ch[i-1](torch.cat(tensors=[skip, out], dim=1))
            else:
                out = self.blocks[i](torch.cat(tensors=[out, x], dim=1))
        return out