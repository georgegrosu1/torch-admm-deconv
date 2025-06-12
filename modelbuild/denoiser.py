import torch
import torch.nn as nn
from modelbuild.blocks import DivergentAttention


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
                 admms: list[dict] = None,
                 denoise: bool = True,
                 frozen: nn.Module = None):
        super(DivergentRestorer, self).__init__()

        self._level_branches = [init_branches]
        for i in range(1, num_levels):
            self._level_branches.append(self._level_branches[-1] * divergence_factor)

        self.frozen = frozen
        self.denoise = denoise
        self.blocks = nn.ModuleList()
        for i in range(num_levels):
            if i == 0:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=in_channels,
                                                      out_channels=filters,
                                                      conv_filters=filters,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=intermediate_activation,
                                                      admms=admms,
                                                      use_varmap=True))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        denoised = self.frozen(x) if self.frozen is not None else x
        out = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](torch.cat(tensors=[out, denoised], dim=1))
        if self.frozen is None:
            return out
        return denoised * out