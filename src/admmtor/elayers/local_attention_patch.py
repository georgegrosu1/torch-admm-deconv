from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from admmtor.elayers.channel_pool import ChannelPool


class AbsLPPool2d(nn.Module):
    """LPPool2d wrapper that takes absolute values before pooling.

    This avoids NaNs that may arise from negative intermediate sums when the
    implementation computes x.pow(p).sum() followed by pow(sum, 1.0/p).
    Using `abs` guarantees the inner quantity is non-negative and prevents
    numerical NaNs for odd `p` values.
    """

    def __init__(self, norm_type: int = 2, kernel_size=1, stride=None, ceil_mode: bool = False, cons: float = 1e-12):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode
        self.cons = cons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.lp_pool2d(x.abs() + self.cons, self.norm_type, self.kernel_size, self.stride, self.ceil_mode)


class PatchProcessor(nn.Module):
    """Applies a learnable residual gate to a flattened patch."""

    def __init__(
        self,
        out_channels: int,
        in_channels: int | None = None,
        features_dim_size: int = 1,
        *,
        downscale_kernel: int = 2,
        downscale_stride: int = 2,
        embedding_dim: int = 64,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        
        if downscale_kernel <= 0:
            raise ValueError("downscale_kernel must be a positive integer")
        if downscale_stride <= 0:
            raise ValueError("downscale_stride must be a positive integer")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features_dim_size = features_dim_size
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.embedding_dim = embedding_dim
        self.spatial_kernel = spatial_kernel
        
        # Build downscale and encoder
        if self.in_channels != out_channels:
            self._init_channel_adapt()
        else:
            self.channel_adapt = nn.Identity()
        if in_channels is None:
            self.in_channels = out_channels
        self._init_downscale()
        self._init_encoder()
        self._init_spatial()
        self._init_attention_weights()
        self.activation = nn.Sigmoid()
        
    def _init_channel_adapt(self) -> None:
        self.channel_adapt = nn.Sequential(
            nn.LazyConv2d(
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
        )
    )
        
    def _init_downscale(self) -> None:
        self.downscale = nn.Sequential(
            nn.LazyConv2d(
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.LazyConv2d(
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            nn.LazyConv2d(
                out_channels=self.out_channels,
                kernel_size=self.downscale_kernel,
                stride=self.downscale_stride,
                bias=False,
            ),
            AbsLPPool2d(norm_type=3, kernel_size=self.downscale_kernel, stride=self.downscale_stride),
        )
        
    def _init_encoder(self) -> None:
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.out_channels * self.features_dim_size),
            nn.GELU(),
            nn.LazyLinear(out_features=self.embedding_dim),
            nn.GELU(),
            nn.LazyLinear(out_features=self.out_channels),
        )
        
    def _init_spatial(self) -> None:
        self.spatial = nn.Sequential(
            nn.LazyConv2d(
                out_channels=self.out_channels, 
                kernel_size=self.spatial_kernel, 
                stride=1, 
                padding=self.spatial_kernel // 2, 
                bias=False),
            nn.LazyInstanceNorm2d(),
            nn.LazyConv2d(
                out_channels=self.out_channels, 
                kernel_size=1, 
                stride=1, 
                bias=True),
            nn.GELU(),
        )
        
    def _init_attention_weights(self) -> None:
        self.alfa_w = nn.Parameter(torch.randn((self.out_channels,)), requires_grad=True)
        self.beta_w = nn.Parameter(torch.randn((self.out_channels,)), requires_grad=True)
        
    def forward_global(self, patch: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = patch.shape
        processed = self.downscale(patch)
        flat = self.encoder(processed)
        gated = self.activation(flat).view(batch, channels, 1, 1)
        return self.activation(self.alfa_w.view(1, -1, 1, 1) * gated.expand(-1, -1, height, width))
    
    def forward_spatial(self, patch: torch.Tensor) -> torch.Tensor:
        spatial_out = self.spatial(patch)
        return self.activation(self.beta_w.view(1, -1, 1, 1) * spatial_out)
    
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        patch = self.channel_adapt(patch)
        glob = self.forward_global(patch)
        spatial = self.forward_spatial(patch)
        return patch * (glob + spatial)


class LocalAttentionPatch(nn.Module):
    """Local attention module that processes spatial patches independently."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        num_processors: int,
        out_channels: int,
        in_channels: int | None = None,
        *,
        features_dim_size: int = 1,
        downscale_kernel: int | tuple[int, int] = 1,
        downscale_stride: int | tuple[int, int] = 1,
        embedding_dim: int = 64,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        if num_processors <= 0:
            raise ValueError("num_processors must be a positive integer")
        if features_dim_size <= 0:
            raise ValueError("features_dim_size must be a positive integer")
        

        self.patch_size = patch_size
        self.stride = stride
        self.num_processors = num_processors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features_dim_size = features_dim_size
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.embedding_dim = embedding_dim
        self.spatial_kernel = spatial_kernel
        self.activation = nn.Sigmoid()
        self.patch_processors = nn.ModuleList()

        self._build_processors()

    def _build_processors(self) -> None:
        if self.patch_processors:
            return
        for _ in range(self.num_processors):
            self.patch_processors.append(
                PatchProcessor(
                    out_channels=self.out_channels,
                    in_channels=self.in_channels,
                    features_dim_size=self.features_dim_size,
                    downscale_kernel=self.downscale_kernel,
                    downscale_stride=self.downscale_stride,
                    embedding_dim=self.embedding_dim,
                    spatial_kernel=self.spatial_kernel,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("LocalAttentionPatch expects input with shape (B, C, H, W)")

        batch, channels, height, width = x.shape
        self._build_processors()

        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        num_patches = patches.shape[-1]
        if num_patches == 0:
            raise ValueError("No patches were extracted; check patch size and stride")
        if num_patches != self.num_processors:
            raise ValueError(
                f"Expected num processors to be same as {num_patches} patches, but got {self.num_processors}"
            )

        patches = patches.reshape(batch, channels, self.patch_size, self.patch_size, -1)
        patches = torch.unbind(patches, dim=-1)

        processed_patches = [
            processor(patch) for processor, patch in zip(self.patch_processors, patches)
        ]

        reconstructed = F.fold(
            torch.stack(processed_patches, dim=-1).reshape(batch, -1, num_patches),
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.stride,
        )

        return self.activation(reconstructed)
    
    
class MultiLAP(nn.Module):
    """Applies multiple LocalAttentionPatch modules."""

    def __init__(
        self,
        patch_sizes: list[int],
        strides: list[int],
        num_processors: list[int],
        unit_out_channels: int,
        in_channels: int | None = None,
        *,
        features_dim_size: int = 1,
        downscale_kernel: int | tuple[int, int] = 1,
        downscale_stride: int | tuple[int, int] = 1,
        embedding_dim: int = 64,
        spatial_kernel: int = 7,
        keep_out_channels: bool = True,
    ) -> None:
        super().__init__()

        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_processors = num_processors
        self.num_modules = len(patch_sizes)
        self.unit_out_channels = unit_out_channels
        self.in_channels = in_channels
        self.features_dim_size = features_dim_size
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.embedding_dim = embedding_dim
        self.spatial_kernel = spatial_kernel
        self.keep_out_channels = keep_out_channels
        self.local_attention_modules = nn.ModuleList()
        
        if self.keep_out_channels:
            self.ch_pool = ChannelPool(
                top_k=unit_out_channels,
                soft=True,
                in_channels=unit_out_channels * self.num_modules,
            )
        else:
            self.ch_pool = nn.Identity()
        

        for patch_size, stride, num_processor in zip(patch_sizes, strides, num_processors):
            self.local_attention_modules.append(
                LocalAttentionPatch(
                    patch_size=patch_size,
                    stride=stride,
                    num_processors=num_processor,
                    out_channels=unit_out_channels,
                    in_channels=in_channels,
                    features_dim_size=features_dim_size,
                    downscale_kernel=downscale_kernel,
                    downscale_stride=downscale_stride,
                    embedding_dim=embedding_dim,
                    spatial_kernel=spatial_kernel,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [module(x) for module in self.local_attention_modules]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.ch_pool(outputs)
        return outputs