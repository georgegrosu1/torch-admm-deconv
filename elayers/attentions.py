import torch
import torch.nn as nn
import torch.nn.functional as torchf

from elayers.varmap import ChannelwiseVariance


def logsumexp_2d(x: torch.Tensor) -> torch.Tensor:
    tensor_flatten = x.view(x.size(0), x.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class BasicConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_activation=True,
                 norm=True,
                 bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.norm = nn.InstanceNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if norm else nn.Identity()
        self.activation = nn.GELU() if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class ChannelPool(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            torch.std(x, 1).unsqueeze(1),
            # torch.max(x, 1)[0].unsqueeze(1),
            # torch.mean(x, 1).unsqueeze(1),
            torch.median(x, 1).values.unsqueeze(1),
            torch.mode(x, 1).values.unsqueeze(1)
        ), dim=1)


class SpatialGate(nn.Module):
    def __init__(self,
                 kernel_size: int = 7,
                 use_activation: bool = False
                 ):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(3, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2,
                                 use_activation=use_activation)

    def forward(self, x: torch.Tensor):
        return x * torchf.sigmoid(self.spatial(self.compress(x)))


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.gate_channels, self.gate_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(self.gate_channels // reduction_ratio, self.gate_channels)
        )
        self.pool_types = pool_types


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_att_sum, pool_out = None, None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool_out = torchf.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                pool_out = torchf.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'lp':
                pool_out = torchf.lp_pool2d(x, 2, (x.size(2), x.size(3)),
                                            stride=(x.size(2), x.size(3)))
            elif pool_type == 'lse':
                # LSE pool only
                pool_out = logsumexp_2d(x)

            channel_att_raw = self.mlp(pool_out)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        return x * torchf.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)


class CBAM(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=('avg', 'max'),
                 use_spatial=False):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.spatial_gate = SpatialGate() if use_spatial else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.channel_gate(x)
        x_out = self.spatial_gate(x_out) if self.spatial_gate is not None else x_out
        return x_out
