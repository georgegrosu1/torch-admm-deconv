import re
import torch
import torch.nn as nn

from admmtor.elayers.admmdeconv import ADMMDeconv
from admmtor.elayers.attentions import CBAM
from admmtor.elayers.channel_pool import ChannelPool
from admmtor.elayers.local_attention_patch import (
    LocalAttentionPatch, 
    MultiLAP
)


@torch.no_grad()
def default_init_weights(
    nn_modules: nn.Module | list[nn.Module], 
    weights_att_names: list[str],
    init_func: callable = nn.init.kaiming_normal_,
    bias_eps: float = 1e-12
    ) -> None:
    nn_modules = nn_modules if isinstance(nn_modules, list) else [nn_modules]
    
    supported_types = re.compile(r'(?i)(?:conv|linear|norm|pool)')
    for nn_module in nn_modules:
        if supported_types.search(nn_module.__class__.__name__):
            for w_name in weights_att_names:
                if getattr(nn_module, w_name) is not None:
                    if 'bias' in w_name:
                        getattr(nn_module, w_name).data.fill_(bias_eps)
                    else:
                        init_func(getattr(nn_module, w_name))


def compute_residual_dec_input_channels(enc_out_channels: list[int], dec_out_channels: list[int]) -> list[int]:
    enc_out_channels_rev = enc_out_channels[::-1]
    return [enc_out_channels_rev[0]] + [enc_out + dec_out for enc_out, dec_out in zip(enc_out_channels_rev[1:],
                                                                              dec_out_channels[:-1])]

def compute_enc_input_channels(in_channels: int, enc_out_channels: list[int],
                               depthwise: bool = False) -> list[int]:
    if depthwise:
        res = [in_channels]
        for i, k in zip(range(len(enc_out_channels)), enc_out_channels):
            res.append(k*res[i])
    return [in_channels] + enc_out_channels[:-1]


def compute_depth_enc_in_out_channels(in_channels: int, enc_out_channels: list[int]) -> tuple[list[int], list[int]]:
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


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class DivergentAttention(nn.Module):
    def __init__(self,
                 branches: int,
                 in_channels: int,
                 out_channels: int,
                 conv_filters: int,
                 gate_channels: int,
                 attention_reduction: int,
                 out_activation: nn.Module = None,
                 admms: list[dict] = None):
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
                                          kernel_size=3))
            self.attentions.append(CBAM(gate_channels=gate_channels, reduction_ratio=attention_reduction,
                                        pool_types=self._pool_types[i%2], use_spatial=True))
            if admms is not None:
                self.admms.append(ADMMDeconv(**admms[i]))

        for conv in self.convs:
            default_init_weights(conv, ['weight', 'bias'])
        default_init_weights(self.convout, ['weight', 'bias'])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
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
                 kernel_size: int | tuple[int, int],
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
        self.chx = nn.Conv2d(in_channels=up_in_ch, out_channels=down_out_ch, kernel_size=1, stride=1,
                                        padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.chx(x)
        x = self.up_block(x)
        x = self.chc(x)
        x = self.down_block(x)
        return res + self.chc2(x)


class LazyMultiReceptiveFieldsConv(nn.Module):
    def __init__(self,
                 out_channels: int,
                 kernel_size: int,
                 rfs: list[int],
        ):
        super(LazyMultiReceptiveFieldsConv, self).__init__()
        self.convs = nn.ModuleList()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.rfs = rfs
        self.pool_out = None # Initialize as None for lazy loading

        for r in rfs:
            self.convs.append(nn.LazyConv2d(out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            dilation=r,
                                            padding='same',
                                            padding_mode='circular',
                                            bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([conv(x) for conv in self.convs], dim=1)

        if self.pool_out is None:
            # Lazy initialization of ChannelPool
            # The in_channels for ChannelPool will be the sum of out_channels from all convolutions
            channel_pool_in_channels = self.out_channels * len(self.convs)
            self.pool_out = ChannelPool(top_k=self.out_channels, soft=True,
                                        differentiable=True, in_channels=channel_pool_in_channels).to(x.device)
            for conv in self.convs:
                default_init_weights(conv, ['weight', 'bias'])

        return self.pool_out(out)
    

class MultiADMM(nn.Module):
    def __init__(self,
                 admm_dicts: list[dict]):
        super(MultiADMM, self).__init__()
        self.admms = nn.ModuleList()
        for admm_dict in admm_dicts:
            self.admms.append(ADMMDeconv(**admm_dict))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([admm_l(x) for admm_l in self.admms], dim=1)


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 activation: nn.Module = None,
                 normalization: nn.Module = None,
                 pool_size: int = 0):
        super(DownBlock, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1, padding=max(0, pool_size-1), padding_mode='zeros', bias=False)
        default_init_weights(self.down_conv, ['weight', 'bias'])

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
                 kernel_size: int | tuple[int, int],
                 activation: nn.Module = None,
                 normalization: nn.Module = None,
                 pool_size: int = 0):
        super(UpBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=1, bias=False)
        default_init_weights(self.up_conv, ['weight', 'bias'])

        self.normalization = normalization
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1) if pool_size != 0 else None
        self.activation = activation


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        x = self.normalization(x) if self.normalization is not None else x
        x = self.activation(x) if self.activation is not None else x
        x = self.max_pool(x) if self.max_pool is not None else x
        return x

