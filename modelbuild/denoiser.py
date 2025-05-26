import torch
import torch.nn as nn

from elayers.admmdeconv import ADMMDeconv
from modelbuild.blocks import DivergentAttention


class DivergentRestorer(nn.Module):
    def __init__(self,
                 branches: list,
                 in_channels: int,
                 final_channels: int,
                 filters: int,
                 gate_channels: int,
                 attention_reduction: int,
                 intermediate_activation: nn.Module = None,
                 output_activation: nn.Module = None,
                 admms: list[dict] = None):
        super(DivergentRestorer, self).__init__()

        self._init_admms(admms)
        if self.admm_blocks is not None:
            self.admm_fusion = nn.Conv2d(in_channels=in_channels * len(admms), out_channels=in_channels, kernel_size=1,
                                         stride=1, padding=0, bias=False)
        else:
            self.admm_fusion = None
        self.branches = branches
        self.blocks = nn.ModuleList()
        num_levels = len(branches)
        for i in range(num_levels):
            if i == 0:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=in_channels if admms is None else in_channels * len(admms),
                                                      out_channels=filters,
                                                      conv_filters=filters,
                                                      enc_depth=1, dec_depth=1, ae_filters=filters//2, ae_kern_size=3,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=intermediate_activation,
                                                      use_varmap=True))
            elif i == num_levels - 1:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=filters + 3,
                                                      out_channels=final_channels,
                                                      conv_filters=filters,
                                                      enc_depth=4, dec_depth=4, ae_filters=filters // 2, ae_kern_size=3,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=output_activation))
            else:
                self.blocks.append(DivergentAttention(branches=self._level_branches[i],
                                                      in_channels=filters + 3,
                                                      out_channels=filters,
                                                      conv_filters=filters,
                                                      enc_depth=3, dec_depth=3, ae_filters=filters // 2, ae_kern_size=2,
                                                      gate_channels=gate_channels,
                                                      attention_reduction=attention_reduction,
                                                      out_activation=intermediate_activation))

    def _init_admms(self, admms: list[dict] | None):
        if admms is None:
            self.admm_blocks = None
        else:
            self.admm_blocks = nn.ModuleList()
            for admm_cfg in admms:
                self.admm_blocks.append(ADMMDeconv(**admm_cfg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.admm_blocks(x) if self.admm_blocks is not None else x
        x = self.admm_fusion if self.admm_fusion is not None else x
        out = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            out = self.blocks[i](torch.cat(tensors=[out, x], dim=1))
        return x * out