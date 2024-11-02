import torch
import torch.nn as nn
from typing import Tuple, List


def compute_residual_dec_input_channels(enc_out_channels: List[int], dec_out_channels: List[int]) -> List[int]:
    enc_out_channels_rev = enc_out_channels[::-1]
    return [enc_out_channels_rev[0]] + [enc_out + dec_out for enc_out, dec_out in zip(enc_out_channels_rev[1:],
                                                                              dec_out_channels[:-1])]


def compute_enc_input_channels(in_channels: int, enc_out_channels: List[int]) -> List[int]:
    return [in_channels] + enc_out_channels[:-1]


class UpDownBock(nn.Module):
    def __init__(self,
                 up_in_ch: int, up_out_ch: int,
                 down_out_ch: int,
                 kernel_size: int | Tuple[int, int],
                 activation: nn.Module = None):
        super(UpDownBock, self).__init__()
        self.up_block = UpBlock(up_in_ch, up_out_ch, kernel_size, activation)
        self.down_block = DownBlock(up_out_ch, down_out_ch, kernel_size, activation)
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_block(x)
        x = self.down_block(x)
        x = self.activation(x) if self.activation is not None else x
        return x


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 activation: nn.Module = None,
                 pool_size: int = 0):
        super(DownBlock, self).__init__()
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1, padding=max(0, pool_size-1), padding_mode='zeros')
        self.normalization = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size != 0 else None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x)
        x = self.normalization(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 activation: nn.Module = None,
                 pool_size: int = 0):
        super(UpBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=1)
        self.normalization = nn.BatchNorm2d(num_features=out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size != 0 else None
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = self.normalization(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x