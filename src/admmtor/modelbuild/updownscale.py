from admmtor.modelbuild.blocks import *


class UpDownScale(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 kernel_sizes: List[int | Tuple[int, int]],
                 activation: nn.Module = None):
        super(UpDownScale, self).__init__()
        assert len(out_channels) == len(kernel_sizes)
        assert len(out_channels) % 2 == 0, 'Module must have even number of blocks'

        # Solve first half
        first_half_out_channels = out_channels[:(len(out_channels)//2)]
        first_half_in_channels = compute_enc_input_channels(in_channels, first_half_out_channels)
        first_half_kernel_sizes = kernel_sizes[:(len(kernel_sizes)//2)]
        self.first_half = self._init_updown_blocks(first_half_in_channels, first_half_out_channels,
                                                   first_half_kernel_sizes, activation)
        # Solve second half
        sec_half_out_channels = out_channels[(len(out_channels) // 2):]
        sec_half_in_channels = compute_residual_dec_input_channels(first_half_out_channels, sec_half_out_channels)
        sec_half_kernel_sizes = kernel_sizes[(len(kernel_sizes) // 2):]
        self.second_half = self._init_updown_blocks(sec_half_in_channels, sec_half_out_channels,
                                                    sec_half_kernel_sizes, activation)


    @staticmethod
    def _init_updown_blocks(in_channels: List[int],
                            out_channels: List[int],
                            kernel_sizes: List[int | Tuple[int, int]],
                            activation: nn.Module = None):
        blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            blocks.append(UpDownBock(in_channels[i], out_channels[i], out_channels[i], kernel_sizes[i], activation))
        return blocks


    def forward_first_half(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for block in self.first_half:
            x = block(x)
            out.append(x)
        return out


    def forward_second_half(self, x: List[torch.Tensor]) -> torch.Tensor:
        x.reverse()
        out = self.second_half[0](x[0])
        for i in range(1, len(x)):
            out = self.second_half[i](torch.cat([x[i], out], dim=1))
        return out


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_second_half(self.forward_first_half(x))

