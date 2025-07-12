import torch
import torch.nn as nn

from enum import Enum


def amedian(x: torch.Tensor) -> torch.Tensor:
    return torch.median(x.flatten().reshape(x.shape[0], x.shape[1], -1), -1).values


def amodes(x: torch.Tensor) -> torch.Tensor:
    return torch.mode(x.flatten().reshape(x.shape[0], x.shape[1], -1), -1).values


def amean(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x.flatten().reshape(x.shape[0], x.shape[1], -1), -1)


def astd(x: torch.Tensor) -> torch.Tensor:
    return torch.std(x.flatten().reshape(x.shape[0], x.shape[1], -1), -1)


def amax(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x.flatten().reshape(x.shape[0], x.shape[1], -1), -1).values


class ChannelCompression(Enum):
    STD = astd
    MEAN = amean
    MAX = amax
    MEDIAN = amedian
    MODE = amodes


class SimpleChannelAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channel_compress_methods: list[ChannelCompression] = (ChannelCompression.STD,
                                                                      ChannelCompression.MEDIAN,
                                                                      ChannelCompression.MODE),):
        super(SimpleChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,
                              stride=1, padding=0, bias=True)
        self.compress_method = channel_compress_methods
        self.compress_weight = nn.Parameter(torch.ones((len(channel_compress_methods), 1)), requires_grad=True)
        self.prob_func = nn.Sigmoid()

    def _get_compressed_vals(self, x: torch.Tensor) -> torch.Tensor:
        compress_vals = torch.stack([compress_method(x).flatten() for compress_method in self.compress_method], dim=-1)
        compress_vals = compress_vals * self.compress_weight
        return torch.sum(compress_vals, dim=0).reshape(x.shape[0], x.shape[1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_compress = self._get_compressed_vals(x)
        return x * self.prob_func(self.conv(x) * weighted_compress)