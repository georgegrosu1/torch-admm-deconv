import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseVariance(nn.Module):
    def __init__(self, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        """
        Custom layer to compute channel-wise variance maps.

        Args:
            kernel_size (int): Size of the kernel (assumed square).
            stride (int): Stride for the sliding window.
            padding (int): Padding to apply to the input.
        """
        super(ChannelwiseVariance, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """
        Compute channel-wise variance maps.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Variance map of shape (B, C, H', W').
        """
        B, C, H, W = x.shape

        # Unfold the input to extract patches of shape (B, C, kernel_size*kernel_size, L)
        patches = F.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )  # Shape: (B, C * kernel_size^2, L)

        # Reshape to (B, C, kernel_size*kernel_size, L)
        patches = patches.view(B, C, self.kernel_size ** 2, -1)

        # Compute mean along patch dimension
        mean = patches.mean(dim=2, keepdim=True)  # Shape: (B, C, 1, L)

        # Compute variance along patch dimension
        variance = ((patches - mean) ** 2).mean(dim=2)  # Shape: (B, C, L)

        # Reshape back to spatial dimensions
        h_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        variance_map = variance.view(B, C, h_out, w_out)

        return variance_map