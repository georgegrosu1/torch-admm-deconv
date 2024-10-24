import torch
import torch.nn as nn
from typing import Tuple


class UpDownBock(nn.Module):
    def __init__(self,
                 kernel_size: int | Tuple[int, int],
                 up_in_ch: int, up_out_ch: int,
                 down_in_ch: int, down_out_ch: int,
                 activation: nn.Module = None):
        super(UpDownBock, self).__init__()
        self.up_block = UpBlock(up_in_ch, up_out_ch, kernel_size, activation)
        self.down_block = DownBlock(down_in_ch, down_out_ch, kernel_size, activation)
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
                 pool_size: int = None):
        super(DownBlock, self).__init__()
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1, padding=pool_size-1, padding_mode='zeros')
        self.normalization = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size is not None else None


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
                 pool_size: int = None):
        super(UpBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=1, padding=pool_size, padding_mode='circular')
        self.normalization = nn.BatchNorm2d(num_features=out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size is not None else None
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = self.normalization(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x