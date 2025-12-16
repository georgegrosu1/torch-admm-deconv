import torch
import torch.nn as nn
from admmtor.modelbuild.blocks import (
    MultiADMM
)
from admmtor.elayers.attentions import CBAM
from admmtor.elayers.local_attention_patch import MultiLAP


class DenoiserV2Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int, 
                 patch_sizes: list[int], 
                 strides: list[int], 
                 num_processors: list[int], 
                 features_dim_size: int = 1, 
                 downscale_kernel: int = 3, 
                 downscale_stride: int = 2, 
                 embedding_dim: int = 16, 
                 spatial_kernel: int = 3,
                 admms_dicts: list[dict] | None = None):
        super(DenoiserV2Block, self).__init__()
        
        self.in_channels = in_channels
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.num_processors = num_processors
        self.out_channels = out_channels
        self.features_dim_size = features_dim_size
        self.downscale_kernel = downscale_kernel
        self.downscale_stride = downscale_stride
        self.embedding_dim = embedding_dim
        self.spatial_kernel = spatial_kernel
        self.admms_dicts = admms_dicts
        
        self._init_resid_up()
        self._init_resid_down()
        self._init_admms()
        self._init_multilap()
        self._init_cbam()
        self._init_postprocess()
        
    def _init_resid_up(self):
        self.resid_u = nn.Sequential(
            nn.LazyConvTranspose2d(self.out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LazyConvTranspose2d(self.out_channels, kernel_size=2, stride=2, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU6(),
            nn.LazyConv2d(self.out_channels, kernel_size=2, stride=2, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ) if self.in_channels != self.out_channels else nn.Identity()
        
    def _init_resid_down(self):
        self.resid_d = nn.Sequential(
            nn.LazyConv2d(self.out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LazyConv2d(self.out_channels, kernel_size=2, stride=2, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU6(),
            nn.LazyConvTranspose2d(self.out_channels, kernel_size=2, stride=2, padding=0, bias=True),
            nn.LazyConvTranspose2d(self.out_channels, kernel_size=3, stride=1, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ) if self.in_channels != self.out_channels else nn.Identity()
        
    def _init_admms(self):
        self.admms = MultiADMM(self.admms_dicts) if self.admms_dicts else nn.Identity()
        
    def _init_postprocess(self):
        self.postprocess = nn.Sequential(
            nn.LazyConvTranspose2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LazyConvTranspose2d(self.out_channels, kernel_size=3, stride=2, padding=0, bias=True),
            nn.GELU(),
            nn.LazyConv2d(self.out_channels, kernel_size=3, stride=2, padding=0, bias=True),
            nn.LazyConv2d(self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
    def _init_multilap(self):
        self.multilap = MultiLAP(
            patch_sizes=self.patch_sizes,
            strides=self.strides,
            num_processors=self.num_processors,
            unit_out_channels=self.out_channels,
            in_channels=self.in_channels,
            features_dim_size=self.features_dim_size,
            downscale_kernel=self.downscale_kernel,
            downscale_stride=self.downscale_stride,
            embedding_dim=self.embedding_dim,
            spatial_kernel=self.spatial_kernel,
            keep_out_channels=True
        )
        
    def _init_cbam(self):
        # Use the block's output channel count for gate_channels so the attention
        # layers match the actual tensor shape returned by MultiLAP.
        # Using `filters` here caused channel mismatches when `out_channels` != `filters`.
        self.cbam = CBAM(gate_channels=self.out_channels, 
                         pool_types=('avg', 'max'), 
                         use_spatial=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid_up = self.resid_u(x)
        resid_down = self.resid_d(x)
        x = self.admms(x)
        x = self.multilap(x)
        x = self.cbam(x)
        x = x + resid_up
        x = self.postprocess(x) + resid_down
        return x


class DenoiserV2(nn.Module):
    def __init__(self,
                 blocks_config: list[dict]):
        super(DenoiserV2, self).__init__()

        self.blocks_config = blocks_config
        self.blocks = nn.ModuleList()
        self.activation = nn.Sigmoid()
        
        self._init_blocks()
        self._init_adapter_x()
        self._init_adapter_out()
        
    def _init_blocks(self) -> None:
        for block_cfg in self.blocks_config:
            block = DenoiserV2Block(**block_cfg)
            self.blocks.append(block)
        
    def _init_adapter_x(self) -> None:
        self.adapter_x = nn.ModuleList()
        for block_cfg in self.blocks_config:
            adapter = nn.Sequential(
                nn.LazyConv2d(block_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=True),
                nn.LazyConv2d(block_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=True),
                nn.LazyConv2d(block_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid()
            )
            self.adapter_x.append(adapter)
            
    def _init_adapter_out(self) -> None:
        self.adapter_out = nn.ModuleList()
        for block_cfg in self.blocks_config:
            adapter = nn.Sequential(
                nn.LazyConv2d(block_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=True),
                nn.LazyConv2d(block_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=True),
                nn.LazyConv2d(block_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            self.adapter_out.append(adapter)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for adapter_x, adapter_out, block in zip(self.adapter_x, self.adapter_out, self.blocks):
            adapted_x = adapter_x(x)
            out = block(out)
            out = adapter_out(out + adapted_x)
        return out
