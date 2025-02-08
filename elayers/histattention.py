import torch
import torch.nn as nn


class PixelFrequencyLayer(nn.Module):
    def __init__(self, num_bins=256):
        """
        Initialize the layer.
        Args:
            num_bins (int): Number of bins for the pixel intensity values (default: 256 for 8-bit images).
        """
        super(PixelFrequencyLayer, self).__init__()
        self.num_bins = num_bins
        self.register_buffer("pixel_probabilities", torch.ones(num_bins) / num_bins)

    def compute_frequencies(self, images):
        """
        Compute pixel intensity frequencies and update probabilities.
        Args:
            images (torch.Tensor): Input images (batch_size, channels, height, width).
        """
        with torch.no_grad():
            # Flatten and compute histogram
            flat_pixels = images.flatten()
            hist = torch.histc(flat_pixels, bins=self.num_bins, min=0, max=self.num_bins - 1)

            # Normalize histogram to probabilities
            total_pixels = flat_pixels.numel()
            self.pixel_probabilities = hist / total_pixels

    def forward(self, images):
        """
        Transform the input image pixels into probabilities.
        Args:
            images (torch.Tensor): Input images (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Transformed images with probabilities.
        """
        # Map pixel values to probabilities
        pixel_indices = images.long()  # Ensure pixel values are integers
        probabilities = self.pixel_probabilities[pixel_indices]
        return probabilities