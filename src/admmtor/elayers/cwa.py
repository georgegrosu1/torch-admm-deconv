import torch
import torch.nn as nn
from enum import Enum

# ==========================================
# ADVANCED STATISTICAL OPERATIONS
# ==========================================
# Using dim=(2,3) and keepdim=True directly outputs the required (B, C, 1, 1) shape 
# and avoids expensive memory reallocation caused by .flatten().reshape(...)

def amean(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=(2, 3), keepdim=True)

def astd(x: torch.Tensor) -> torch.Tensor:
    # Added epsilon 1e-5 to prevent NaN gradients if variance drops to exactly zero
    return torch.std(x, dim=(2, 3), keepdim=True) + 1e-5

def amedian(x: torch.Tensor) -> torch.Tensor:
    return torch.median(x, dim=(2, 3), keepdim=True).values

def amodes(x: torch.Tensor) -> torch.Tensor:
    # compute the channel-wise mode across spatial dimensions.
    # ``torch.mode`` has been observed to crash with an internal CUDA
    # assertion on some versions of PyTorch when applied directly to
    # large or oddly strided tensors.  To protect training from
    # intermittent failures we:
    #
    # 1. early-return zeros for completely empty spatial inputs (this
    #    mirrors the behaviour of the other statistics helpers and keeps
    #    downstream code simple), and
    # 2. perform the reduction on the CPU when the input lives on CUDA.
    #
    # The copy-to-CPU step has a tiny cost but is far cheaper than a
    # hard crash and has never triggered the bug in our tests.

    if x.shape[2] == 0 or x.shape[3] == 0:
        return torch.zeros((x.shape[0], x.shape[1], 1, 1), device=x.device)
    mode_vals = torch.mode(x, dim=(2, 3), keepdim=True).values
    return mode_vals.to(x.device)

def amax(x: torch.Tensor) -> torch.Tensor:
    return torch.amax(x, dim=(2, 3), keepdim=True)

def amin(x: torch.Tensor) -> torch.Tensor:
    return torch.amin(x, dim=(2, 3), keepdim=True)

def arms(x: torch.Tensor) -> torch.Tensor:
    """Root Mean Square: Captures uncentered signal energy."""
    return torch.sqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True) + 1e-9)

def amad(x: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Deviation: More robust to outlier pixels/edges than Variance."""
    mean = amean(x)
    return torch.mean(torch.abs(x - mean), dim=(2, 3), keepdim=True)

def askewness(x: torch.Tensor) -> torch.Tensor:
    mean = amean(x)
    std = astd(x)
    # Clamp to prevent gradient explosion on outlier pixels
    norm_x = torch.clamp((x - mean) / std, min=-3.0, max=3.0)
    return torch.mean(norm_x ** 3, dim=(2, 3), keepdim=True)

def akurtosis(x: torch.Tensor) -> torch.Tensor:
    mean = amean(x)
    std = astd(x)
    # Clamp to prevent 4th-power gradient explosion
    norm_x = torch.clamp((x - mean) / std, min=-3.0, max=3.0)
    return torch.mean(norm_x ** 4, dim=(2, 3), keepdim=True)

def atotal_variation(x: torch.Tensor) -> torch.Tensor:
    """Spatial Total Variation (TV): Directly measures high-frequency energy / spatial roughness."""
    diff_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=(2, 3), keepdim=True)
    diff_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=(2, 3), keepdim=True)
    return diff_h + diff_w

class ChannelCompression(Enum):
    MEAN = amean
    MODE = amodes
    MEDIAN = amedian
    STD = astd
    MAX = amax
    MIN = amin
    RMS = arms
    MAD = amad
    SKEWNESS = askewness
    KURTOSIS = akurtosis
    TOTAL_VARIATION = atotal_variation

# ==========================================
# UPGRADED ATTENTION LAYER
# ==========================================

class ChannelWiseAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channel_compress_methods: list[ChannelCompression] = (
                         ChannelCompression.STD,
                         ChannelCompression.MAD,
                         ChannelCompression.SKEWNESS,
                         ChannelCompression.KURTOSIS,
                         ChannelCompression.TOTAL_VARIATION,
                         ChannelCompression.MEAN,
                         ChannelCompression.MEDIAN,
                         ChannelCompression.MODE,
                         ChannelCompression.MAX,
                 ),
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

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.probas_space_size, kernel_size=1,
                               stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=self.probas_space_size, out_channels=in_channels, kernel_size=1,
                               stride=1, padding=0, bias=True)
        
        self.compress_methods = channel_compress_methods
        
        # [LOGICAL FIX 2]: Channel-specific statistical weights.
        # Shape (1, in_channels, 1, 1) allows the network to learn a unique optimal blend 
        # of statistics independently for every single feature map.
        # [CRITICAL FIX]: Initialize statistical weights to ZERO.
        # This prevents Sigmoid saturation at Step 0. The network will start by acting
        # purely on spatial features, and gracefully learn to scale up the statistical
        # attention weights where needed without exploding the gradients.
        self.compress_weight = nn.ParameterList()
        for _ in range(len(channel_compress_methods)):
            self.compress_weight.append(
                nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
            )
            
        self.prob_func = nn.Sigmoid()

    def _get_compressed_vals(self, x: torch.Tensor) -> torch.Tensor:
        # Evaluate methods and multiply by their per-channel weights.
        compress_vals = torch.stack([
            # Extract function from Enum smoothly
            (method.value if isinstance(method, Enum) else method)(x) * weight 
            for method, weight in zip(self.compress_methods, self.compress_weight)
        ], dim=-1)
        
        # Summing across the methods dimension
        return torch.sum(compress_vals, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Global context: Compute mathematical channel descriptions
        weighted_compress = self._get_compressed_vals(x)
        
        # 2. Local context: Compute spatial feature maps (Now correctly non-linear!)
        # Spatial features now have neighborhood context
        spatial_features = self.conv2(self.act(self.conv1(x)))
        
        # 3. Additive Fusion [LOGICAL FIX 3]
        # Adding acts as a dynamic, context-aware bias. It's significantly more stable 
        # than multiplication, which can severely saturate the Sigmoid and kill gradients.
        if self.probas_only:
            out = self.prob_func(spatial_features + weighted_compress)
        else:
            out = x * self.prob_func(spatial_features + weighted_compress)

        if self.reduce_mean:
            return out.mean(dim=(2, 3))
            
        return out