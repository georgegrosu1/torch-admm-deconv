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


def amin(x: torch.Tensor) -> torch.Tensor:
    return torch.min(x.flatten().reshape(x.shape[0], x.shape[1], -1), -1).values


class ChannelCompression(Enum):
    STD = astd
    MEAN = amean
    MAX = amax
    MEDIAN = amedian
    MODE = amodes
    MIN = amin


class ChannelWiseAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channel_compress_methods: list[ChannelCompression] = (ChannelCompression.STD,
                                                                      ChannelCompression.MEDIAN, 
                                                                      ChannelCompression.MODE,
                                                                       ChannelCompression.MAX,
                                                                       ChannelCompression.MEAN),
                 probas_ch_factor: int = 2,
                 compress_judges_mult: int = 10,
                 reduce_probas_space: bool = False,
                 reduce_mean: bool = False,
                 probas_only: bool = False):
        super(ChannelWiseAttention, self).__init__()
        self.in_channels = in_channels
        self.probas_ch_factor = probas_ch_factor
        self.reduce_probas_space = reduce_probas_space
        self.reduce_mean = reduce_mean
        self.probas_only = probas_only
        self.compress_judges_mult = compress_judges_mult

        self.probas_space_size = in_channels // probas_ch_factor if reduce_probas_space else in_channels * probas_ch_factor

        # self.upscale = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.probas_space_size, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.probas_space_size, out_channels=in_channels, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.compress_methods = channel_compress_methods
        self.compress_weight = nn.ParameterList()
        for _ in range(len(channel_compress_methods)):
            self.compress_weight.append(nn.Parameter(torch.ones((1,)), requires_grad=True))
        self.prob_func = nn.Sigmoid()

    def _get_compressed_vals(self, x: torch.Tensor) -> torch.Tensor:
        compress_vals = torch.stack([compress_method(x) * compress_weight
                                     for compress_method, compress_weight in
                                     zip(self.compress_methods, self.compress_weight)], dim=-1)
        return torch.sum(compress_vals, dim=-1).reshape(x.shape[0], x.shape[1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_compress = self._get_compressed_vals(x)
        if self.probas_only:
            out = self.prob_func(self.conv2(self.conv1(x)) * weighted_compress)
        else:
            out = x * self.prob_func(self.conv2(self.conv1(x)) * weighted_compress)

        if self.reduce_mean:
            return out.mean(dim=(2, 3))
        return out