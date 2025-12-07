from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class PatchProcessor(nn.Module):
    """Applies a learnable residual gate to a flattened patch."""

    def __init__(
        self,
        out_channels: int,
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
        
        self.out_channels = out_channels
        self.features_dim_size = features_dim_size
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.embedding_dim = embedding_dim
        self.spatial_kernel = spatial_kernel
        
        # Build downscale and encoder
        self._init_downscale()
        self._init_encoder()
        self._init_spatial()
        self._init_attention_weights()
        self.activation = nn.Sigmoid()
        
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
                out_channels=self.features_dim_size,
                kernel_size=self.downscale_kernel,
                stride=self.downscale_stride,
                bias=False,
            ),
            nn.LPPool2d(norm_type=3, kernel_size=self.downscale_kernel, stride=self.downscale_stride),
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
                bias=True),
            nn.LazyInstanceNorm2d(),
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
        return nn.functional.sigmoid(self.alfa_w.view(1, -1, 1, 1)) * gated.expand(-1, -1, height, width)
    
    def forward_spatial(self, patch: torch.Tensor) -> torch.Tensor:
        spatial_out = self.spatial(patch)
        return nn.functional.sigmoid(self.beta_w.view(1, -1, 1, 1)) * spatial_out
    
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return patch * (self.forward_global(patch) + self.forward_spatial(patch))


class LocalAttentionPatch(nn.Module):
    """Local attention module that processes spatial patches independently."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        num_processors: int,
        *,
        channels: Optional[int] = None,
        features_multiplier: int = 1,
        downscale_kernel: int | tuple[int, int] = 1,
        downscale_stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        if num_processors <= 0:
            raise ValueError("num_processors must be a positive integer")
        if features_multiplier <= 0:
            raise ValueError("features_multiplier must be a positive integer")
        

        self.patch_size = patch_size
        self.stride = stride
        self.num_processors = num_processors
        self.in_channels: Optional[int] = channels
        self.features_multiplier = features_multiplier
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.patch_processors = nn.ModuleList()

        if channels is not None:
            self._build_processors(channels)

    def _build_processors(self, channels: int) -> None:
        if self.patch_processors:
            return
        self.in_channels = channels
        for _ in range(self.num_processors):
            self.patch_processors.append(
                PatchProcessor(
                    channels,
                    self.features_multiplier,
                    downscale_kernel=self.downscale_kernel,
                    downscale_stride=self.downscale_stride,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("LocalAttentionPatch expects input with shape (B, C, H, W)")

        batch, channels, height, width = x.shape
        if self.in_channels is None:
            self._build_processors(channels)
        elif channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, received {channels}"
            )

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

        return reconstructed
