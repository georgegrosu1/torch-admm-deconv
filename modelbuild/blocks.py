import torch
import torch.nn as nn
from typing import Tuple, List

from elayers.admmdeconv import ADMMDeconv
from elayers.attentions import CBAM


def compute_residual_dec_input_channels(enc_out_channels: List[int], dec_out_channels: List[int]) -> List[int]:
    enc_out_channels_rev = enc_out_channels[::-1]
    return [enc_out_channels_rev[0]] + [enc_out + dec_out for enc_out, dec_out in zip(enc_out_channels_rev[1:],
                                                                              dec_out_channels[:-1])]


def compute_enc_input_channels(in_channels: int, enc_out_channels: List[int],
                               depthwise: bool = False) -> List[int]:
    if depthwise:
        res = [in_channels]
        for i, k in zip(range(len(enc_out_channels)), enc_out_channels):
            res.append(k*res[i])
    return [in_channels] + enc_out_channels[:-1]


def compute_depth_enc_in_out_channels(in_channels: int, enc_out_channels: List[int]) -> tuple[list[int], list[int]]:
    res = [in_channels]
    for i, k in zip(range(len(enc_out_channels)), enc_out_channels):
        res.append(k * res[i])
    ins, outs = res[:-1], res[1:]
    return ins, outs


def conv2d_pooling_output_shape(
    input_shape,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    pooling_size=None,
    pooling_stride=None,
    pooling_padding=0
) -> tuple[int, int]:
    """
    Computes the output shape after a Conv2d layer and an optional pooling layer.

    Parameters:
    - input_shape (tuple): The input shape as (height, width).
    - kernel_size (int or tuple): The size of the kernel/filter (height, width).
    - stride (int or tuple): The stride of the convolution (height, width).
    - padding (int or tuple): The padding applied to the input (height, width).
    - dilation (int or tuple): The dilation factor (height, width).
    - pooling_size (int or tuple): The size of the pooling filter (height, width).
    - pooling_stride (int or tuple): The stride of the pooling operation (height, width). Defaults to pooling_size if not specified.
    - pooling_padding (int or tuple): The padding applied to the input for pooling (height, width).

    Returns:
    - tuple: The output shape as (out_height, out_width).
    """
    # Ensure convolution parameters are tuples
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Extract convolution dimensions
    in_height, in_width = input_shape
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = stride
    padding_height, padding_width = padding
    dilation_height, dilation_width = dilation

    # Compute output dimensions after Conv2d
    out_height = ((in_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height) + 1
    out_width = ((in_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width) + 1

    # If pooling is specified, compute the output shape after pooling
    if pooling_size is not None:
        if isinstance(pooling_size, int):
            pooling_size = (pooling_size, pooling_size)
        if pooling_stride is None:
            pooling_stride = pooling_size
        if isinstance(pooling_stride, int):
            pooling_stride = (pooling_stride, pooling_stride)
        if isinstance(pooling_padding, int):
            pooling_padding = (pooling_padding, pooling_padding)

        pool_height, pool_width = pooling_size
        pool_stride_height, pool_stride_width = pooling_stride
        pool_padding_height, pool_padding_width = pooling_padding

        # Compute output dimensions after pooling
        out_height = ((out_height + 2 * pool_padding_height - pool_height) // pool_stride_height) + 1
        out_width = ((out_width + 2 * pool_padding_width - pool_width) // pool_stride_width) + 1

    return out_height, out_width


class DivergentAttention(nn.Module):
    def __init__(self,
                 branches: int,
                 in_channels: int,
                 out_channels: int,
                 conv_filters: int,
                 gate_channels: int,
                 attention_reduction: int,
                 out_activation: nn.Module = None,
                 admms: list[dict] = None,
                 use_varmap: bool = False):
        super(DivergentAttention, self).__init__()

        if admms is not None:
            assert len(admms) == branches

        self._pool_types = [('avg', 'max'), ('lp', 'lse')]
        self.admms = nn.ModuleList() if admms is not None else None
        self.out_activation = out_activation if out_activation is not None else nn.Identity()
        self.convs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.convout = nn.Conv2d(in_channels=conv_filters*branches, out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=0, bias=True)
        for i in range(branches):
            self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=conv_filters, kernel_size=1, stride=1,
                                        padding=0, bias=True))
            self.convs.append(UpDownBlock(up_in_ch=in_channels, up_out_ch=in_channels, down_out_ch=conv_filters,
                                          kernel_size=2))
            self.attentions.append(CBAM(gate_channels=gate_channels, reduction_ratio=attention_reduction,
                                        pool_types=self._pool_types[i%2], use_spatial=True, use_varmap=use_varmap))
            if admms is not None:
                self.admms.append(ADMMDeconv(**admms[i]))

        for conv in self.convs:
            default_init_weights(conv)
        default_init_weights(self.convout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.admms is not None:
            outs = [conv(admm(x)) for conv, admm in zip(self.convs, self.admms)]
        else:
            outs = [conv(x) for conv in self.convs]
        outs_a = torch.cat(tensors=[attention(feat) + feat for attention, feat in
                                    zip(self.attentions[:len(self.attentions) // 2], outs[:len(outs) // 2])], dim=1)
        outs_b = torch.cat(tensors=[attention(feat) + feat for attention, feat in
                                    zip(self.attentions[len(self.attentions) // 2:], outs[len(outs) // 2:])], dim=1)
        outs = torch.cat([outs_a * outs_b, outs_a + outs_b], dim=1)
        return self.out_activation(self.convout(outs))


class UpDownBlock(nn.Module):
    def __init__(self,
                 up_in_ch: int, up_out_ch: int,
                 down_out_ch: int,
                 kernel_size: int | Tuple[int, int],
                 activation: nn.Module = None,
                 normalization: nn.Module = None,
                 pool_size: int = 0):
        super(UpDownBlock, self).__init__()
        self.up_block = UpBlock(up_in_ch, up_out_ch, kernel_size, normalization, activation, pool_size)
        self.down_block = DownBlock(up_out_ch, down_out_ch, kernel_size, normalization, activation, pool_size)
        self.chc = nn.Conv2d(in_channels=up_out_ch, out_channels=up_out_ch, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.chc2 = nn.Conv2d(in_channels=down_out_ch, out_channels=down_out_ch, kernel_size=1, stride=1,
                             padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_block(x)
        x = self.chc(x)
        x = self.down_block(x)
        return self.chc2(x)


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 activation: nn.Module = None,
                 normalization: nn.Module = None,
                 pool_size: int = 0):
        super(DownBlock, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1, padding=max(0, pool_size-1), padding_mode='zeros', bias=False)
        default_init_weights(self.down_conv)

        self.normalization = normalization
        self.activation = activation
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size != 0 else None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down_conv(x)
        x = self.normalization(x) if self.normalization is not None else x
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 activation: nn.Module = None,
                 normalization: nn.Module = None,
                 pool_size: int = 0):
        super(UpBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=1, bias=False)
        default_init_weights(self.up_conv)

        self.normalization = normalization
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size != 0 else None
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = self.normalization(x) if self.normalization is not None else x
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x


class DepthwiseDownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 activation: nn.Module = None,
                 pool_size: int = 0,
                 use_bias: bool = True):
        super(DepthwiseDownBlock, self).__init__()

        print(out_channels, in_channels)

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, padding=max(0, pool_size-1),
                                    padding_mode='zeros', bias=use_bias, groups=in_channels)
        default_init_weights(self.depth_conv, {'a': 0, 'mode': 'fan_in', 'nonlinearity': 'relu'})

        self.activation = activation
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size != 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x


@torch.no_grad()
def default_init_weights(nn_modules: nn.Module | list[nn.Module]):
    nn_modules = nn_modules if isinstance(nn_modules, list) else [nn_modules]
    for nn_module in nn_modules:
        if isinstance(nn_module, nn.Conv2d) or isinstance(nn_module, nn.ConvTranspose2d):
            nn.init.xavier_normal_(nn_module.weight)
            if nn_module.bias is not None:
                nn_module.bias.data.fill_(0)