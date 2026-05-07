from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from admmtor.elayers.channel_pool import ChannelPool
from admmtor.elayers.stats_pool import GeoMeanPool2d
from admmtor.elayers.gating import GeometricGating
from admmtor.modelbuild.weights_init import default_init_weights
    

class PatchProcessor(nn.Module):
    """Applies a learnable residual gate to a flattened patch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int = 64,
        *,
        downscale_levels: int = 2,
        downscale_kernel: int = 2,
        downscale_stride: int = 2,
        spatial_kernel: int = 5,
        spatial_dilation: int = 2,
    ) -> None:
        super().__init__()
        
        if downscale_kernel <= 0:
            raise ValueError("downscale_kernel must be a positive integer")
        if downscale_stride <= 0:
            raise ValueError("downscale_stride must be a positive integer")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.downscale_levels = downscale_levels
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.spatial_kernel = spatial_kernel
        self.spatial_dilation = spatial_dilation
        
        if in_channels is None:
            self.in_channels = out_channels
        if self.in_channels != out_channels:
            self._init_channel_adapt()
        else:
            self.channel_adapt = nn.Identity()
        self._init_downscale()
        self._init_encoder()
        self._init_spatial()
        self._init_attention_weights()
        self.activation = nn.Sigmoid()
        
        # Initialize weights
        default_init_weights(self.modules())
        
    def _init_channel_adapt(self) -> None:
        self.channel_adapt =  nn.LazyConv2d(
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        
    def _init_downscale(self) -> None:
        downscale = nn.ModuleList()
        for _ in range(self.downscale_levels):
            downscale.append(
                nn.LazyConv2d(
                    out_channels=self.out_channels,
                    kernel_size=1,
                    bias=True,
                )
            )
            downscale.append(
                GeoMeanPool2d(kernel_size=self.downscale_kernel, stride=self.downscale_stride)
            )
        self.downscale = nn.Sequential(
            *downscale,
        )
        
    def _init_encoder(self) -> None:
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.embedding_dim),
            GeometricGating(),
            nn.LazyLinear(out_features=self.out_channels),
            nn.SiLU(inplace=True),
        )
        
    def _init_spatial(self) -> None:
        self.spatial = nn.Sequential(
            nn.LazyConv2d(
                out_channels=self.out_channels, 
                kernel_size=self.spatial_kernel, 
                dilation=self.spatial_dilation,
                padding='same',
                padding_mode='circular',
                bias=False),
            nn.LazyConv2d(
                out_channels=self.out_channels, 
                kernel_size=1, 
                bias=True),
            nn.ReLU(inplace=True),
        )
        
    def _init_attention_weights(self) -> None:
        self.alfa_w = nn.Parameter(torch.zeros((1, self.out_channels, 1, 1)), requires_grad=True)
        self.beta_w = nn.Parameter(torch.zeros((1, self.out_channels, 1, 1)), requires_grad=True)
        self.gamma_w = nn.Parameter(torch.zeros((1, self.out_channels, 1, 1)), requires_grad=True)
        
    def forward_global(self, patch: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = patch.shape
        processed = self.downscale(patch)
        flat = self.encoder(processed)
        gated = flat.view(batch, channels, 1, 1)
        return patch * self.activation(gated.expand(-1, -1, height, width))
    
    def forward_spatial(self, patch: torch.Tensor) -> torch.Tensor:
        spatial_out = self.spatial(patch)
        return patch * self.activation(spatial_out)
    
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        patch = self.channel_adapt(patch)
        glob = self.forward_global(patch) * self.alfa_w
        patch = patch + glob
        spatial = self.forward_spatial(patch) * self.beta_w
        patch = patch + spatial
        return patch + self.activation(glob + spatial) * self.gamma_w


class LocalAttentionPatch(nn.Module):
    """Local attention module that processes spatial patches independently."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        stride: int | None = None,
        *,
        embedding_dim: int = 1,
        downscale_levels: int = 2,
        downscale_kernel: int | tuple[int, int] = 1,
        downscale_stride: int | tuple[int, int] = 1,
        spatial_kernel: int = 5,
        spatial_dilation: int = 2,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        self.patch_size = patch_size
        self.stride = patch_size if stride is None else stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.downscale_levels = downscale_levels
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.spatial_kernel = spatial_kernel
        self.spatial_dilation = spatial_dilation
        self.activation = nn.Sigmoid()
        self.patch_processor = self._build_processor()

    def _build_processor(self):
        return PatchProcessor(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                embedding_dim=self.embedding_dim,
                downscale_levels=self.downscale_levels,
                downscale_kernel=self.downscale_kernel,
                downscale_stride=self.downscale_stride,
                spatial_kernel=self.spatial_kernel,
                spatial_dilation=self.spatial_dilation,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("LocalAttentionPatch expects input with shape (B, C, H, W)")

        batch, channels, height, width = x.shape

        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        num_patches = patches.shape[-1]
        if num_patches == 0:
            raise ValueError("No patches were extracted; check patch size and stride")

        patches = patches.reshape(batch, channels, self.patch_size, self.patch_size, -1)
        patches = torch.unbind(patches, dim=-1)

        processed_patches = [
            self.patch_processor(patch) for patch in patches
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
        num_processors: list[int],
        unit_out_channels: int,
        in_channels: int | None = None,
        *,
        embedding_dim: int = 64,
        downscale_kernel: int | tuple[int, int] = 1,
        downscale_stride: int | tuple[int, int] = 1,
        spatial_kernel: int = 7,
        keep_out_channels: bool = True,
    ) -> None:
        super().__init__()

        self.patch_sizes = patch_sizes
        self.strides = patch_sizes
        self.num_processors = num_processors
        self.num_modules = len(patch_sizes)
        self.unit_out_channels = unit_out_channels
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
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
        

        for patch_size, num_processor in zip(patch_sizes, num_processors):
            self.local_attention_modules.append(
                LocalAttentionPatch(
                    patch_size=patch_size,
                    num_processors=num_processor,
                    out_channels=unit_out_channels,
                    in_channels=in_channels,
                    embedding_dim=embedding_dim,
                    downscale_kernel=downscale_kernel,
                    downscale_stride=downscale_stride,
                    spatial_kernel=spatial_kernel,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [module(x) for module in self.local_attention_modules]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.ch_pool(outputs)
        return outputs