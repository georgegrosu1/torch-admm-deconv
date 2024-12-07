import torch
import torch.nn as nn
from typing import Tuple, List


def compute_residual_dec_input_channels(enc_out_channels: List[int], dec_out_channels: List[int]) -> List[int]:
    enc_out_channels_rev = enc_out_channels[::-1]
    return [enc_out_channels_rev[0]] + [enc_out + dec_out for enc_out, dec_out in zip(enc_out_channels_rev[1:],
                                                                              dec_out_channels[:-1])]


def compute_enc_input_channels(in_channels: int, enc_out_channels: List[int]) -> List[int]:
    return [in_channels] + enc_out_channels[:-1]


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


def relu1(x: torch.Tensor) -> torch.Tensor:
    return torch.min(torch.tensor([1], dtype=x.dtype, device=x.device), torch.relu(x))


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
        self.normalization = nn.InstanceNorm2d(num_features=out_channels, affine=True)
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
        self.normalization = nn.InstanceNorm2d(num_features=out_channels, affine=True)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size != 0 else None
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = self.normalization(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x