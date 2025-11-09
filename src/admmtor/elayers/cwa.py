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
        
        self.ca1 = nn.LazyLinear(self.probas_space_size * len(channel_compress_methods), bias=True)
        self.ca2 = nn.ModuleList([nn.LazyConv1d(self.probas_space_size, kernel_size=1, bias=True) for _ in range(len(channel_compress_methods))])
        self.ca3 = nn.LazyLinear(1, bias=True)
        self.ca4 = nn.LazyConv1d(in_channels, kernel_size=1, bias=True)
        
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

        # Apply ca1 ca2 ca3
        ca_out = self.ca1(weighted_compress.reshape(x.shape[0], x.shape[1]))
        ca_out = ca_out.reshape(x.shape[0], -1, 1)
        ca_out = torch.cat([self.ca2[i](ca_out) for i in range(len(self.compress_methods))], dim=1)
        ca_out = self.ca4(self.ca3(ca_out))
        ca_out = ca_out.reshape(weighted_compress.shape)
        
        if self.probas_only:
            out = self.prob_func(self.conv2(self.conv1(x)) * ca_out)
        else:
            out = x * self.prob_func(self.conv2(self.conv1(x)) * ca_out)

        if self.reduce_mean:
            return out.mean(dim=(2, 3))
        return out