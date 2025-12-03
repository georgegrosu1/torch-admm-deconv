from typing import List

from admmtor.modelbuild.blocks import *


class AEModule(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: List[int | Tuple[int, int]],
                 activation: nn.Module = None,
                 pool_size: int = 0):
        super(AEModule, self).__init__()

        assert len(in_channels) == len(out_channels) == len(kernel_sizes), \
            'in_channels, out_channels, and kernel_sizes must have same lengths'
        self._init_blocks(in_channels, out_channels, kernel_sizes, activation, pool_size)


    def _init_blocks(self,
                     in_channels: List[int],
                     out_channels: List[int],
                     kernel_sizes: List[int | Tuple[int, int]],
                     activation: nn.Module = None,
                     pool_size: int = 0):
        raise NotImplementedError


    def forward(self, x: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Encoder(AEModule):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: List[int | Tuple[int, int]],
                 activation: nn.Module = None,
                 pool_size: int = 0):
        super(Encoder, self).__init__(in_channels, out_channels, kernel_sizes, activation, pool_size)


    def _init_blocks(self,
                     in_channels: List[int],
                     out_channels: List[int],
                     kernel_sizes: List[int | Tuple[int, int]],
                     activation: nn.Module = None,
                     pool_size: int = 0):
        blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            blocks.append(DownBlock(in_channels[i], out_channels[i], kernel_sizes[i], activation, pool_size))
        self.blocks = blocks


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out


class Decoder(AEModule):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 kernel_sizes: List[int | Tuple[int, int]],
                 activation: nn.Module = None,
                 pool_size: int = 0):
        super(Decoder, self).__init__(in_channels, out_channels, kernel_sizes, activation, pool_size)


    def _init_blocks(self,
                     in_channels: List[int],
                     out_channels: List[int],
                     kernel_sizes: List[int | Tuple[int, int]],
                     activation: nn.Module = None,
                     pool_size: int = 0):
        blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            blocks.append(UpBlock(in_channels[i], out_channels[i], kernel_sizes[i], activation, pool_size))
        self.blocks = blocks


    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        x.reverse()
        out = self.blocks[0](x[0])
        for i in range(1, len(x)):
            out = self.blocks[i](torch.cat([x[i], out], dim=1))
        return out


class Autoencoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 enc_out_channels: List[int],
                 dec_out_channels: List[int],
                 kernel_sizes: List[int | Tuple[int, int]],
                 activation: nn.Module = None,
                 pool_size: int = 0):
        super(Autoencoder, self).__init__()
        # Solve Encoder
        enc_in_channels = compute_enc_input_channels(in_channels, enc_out_channels)
        self.encoder = Encoder(enc_in_channels, enc_out_channels, kernel_sizes, activation, pool_size)
        # Solve decoder
        dec_kernel_sizes = kernel_sizes[::-1]
        dec_in_channels = compute_residual_dec_input_channels(enc_out_channels, dec_out_channels)
        self.decoder = Decoder(dec_in_channels, dec_out_channels, dec_kernel_sizes, activation, pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
