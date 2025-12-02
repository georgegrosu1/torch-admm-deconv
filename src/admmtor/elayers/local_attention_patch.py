from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class PatchProcessor(nn.Module):
    """Applies a learnable residual gate to a flattened patch."""

    def __init__(
        self,
        channels: int,
        features_multiplier: int = 1,
        *,
        downscale_kernel: int | tuple[int, int] = 1,
        downscale_stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        
        if isinstance(downscale_kernel, tuple):
            if any(k <= 0 for k in downscale_kernel):
                raise ValueError("downscale_kernel entries must be positive")
        elif downscale_kernel <= 0:
            raise ValueError("downscale_kernel must be a positive integer")
        if isinstance(downscale_stride, tuple):
            if any(s <= 0 for s in downscale_stride):
                raise ValueError("downscale_stride entries must be positive")
        elif downscale_stride <= 0:
            raise ValueError("downscale_stride must be a positive integer")
        
        self.channels = channels
        self.features_multiplier = features_multiplier
        self.downscale = nn.LazyConv2d(
            out_channels=channels,
            kernel_size=downscale_kernel,
            stride=downscale_stride,
        )
        self.linear = nn.LazyLinear(out_features=channels * features_multiplier)
        self.conv1d_a_1 = nn.LazyConv1d(out_channels=channels, kernel_size=features_multiplier)
        self.conv1d_a_2 = nn.LazyConv1d(out_channels=channels, kernel_size=1, bias=True)
        self.conv2d_b_1 = nn.LazyConvTranspose2d(out_channels=channels, kernel_size=5, bias=True)
        self.conv2d_b_2 = nn.LazyConv2d(out_channels=channels, kernel_size=1, bias=True)
        self.conv2d_b_3 = nn.LazyConv2d(out_channels=channels, kernel_size=5, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = patch.shape
        processed = self.downscale(patch)
        flat = processed.view(batch, -1)
        gated = self.linear(flat)
        gated = self.conv1d_a_1(gated.view(batch, -1, self.features_multiplier))
        gated = self.conv1d_a_2(gated)
        gated = self.activation(gated).view(batch, channels, 1, 1)

        res = self.conv2d_b_1(patch)
        res = self.conv2d_b_2(res)
        res = self.conv2d_b_3(res)
        return patch + res * gated.expand(-1, -1, height, width)


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
