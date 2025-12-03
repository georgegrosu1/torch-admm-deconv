import torch
import torch.nn as nn


class ChannelPool(nn.Module):
    """Select the top-K channels per sample using the maximum activation over spatial dims.

    Params
    - top_k: int - How many channels to keep per sample

    Behavior
    - Input: Tensor of shape (B, C, H, W)
    - Computes per-sample per-channel score using the maximum absolute activation over H and W
    - Selects the top-K channels (per sample) based on that score
    - Returns Tensor of shape (B, K, H, W)

        Hard vs. soft selection:
        - By default (soft=False) the top-K channels are selected using hard indexing (non-differentiable)
            and returned as-is.
        - If soft=True, the layer computes per-channel probabilities via softmax(scores/temperature) and
            scales the selected channels by their corresponding probabilities (optionally normalized across the
            selected K indices) before returning them. Note that the indices returned by top-k are still hard
            (non-differentiable); however, the scaling is differentiable with respect to the input scores.
    """

    def __init__(self, 
                 top_k: int, 
                 soft: bool = False, 
                 temperature: float = 1.0, 
                 normalize_weights: bool = True, 
                 differentiable: bool = False, 
                 in_channels: int | None = None,
                 const: float = 1e-12):
        super(ChannelPool, self).__init__()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = int(top_k)
        self.soft = bool(soft)
        self.temperature = float(temperature)
        self.normalize_weights = bool(normalize_weights)
        self.differentiable = bool(differentiable)
        self.in_channels = in_channels

        # If differentiable mode is active we need weights for each head (K) over channels (C)
        if self.differentiable:
            if self.in_channels is None:
                raise ValueError("in_channels must be provided when differentiable=True")
            C = int(self.in_channels)
            K = int(self.top_k)
            # Parameter shape (K, C) used to compute attention logits per head
            self.weights_param = nn.Parameter(torch.empty(K, C))
            nn.init.xavier_uniform_(self.weights_param)
        else:
            self.register_parameter('weights_param', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, C, H, W) tensor

        Returns:
            selected_feature_maps: (B, K, H, W) tensor with the top-K channels per sample
        """
        if x.dim() != 4:
            raise ValueError("input tensor must be 4D with shape (B, C, H, W)")

        B, C, H, W = x.shape

        if self.top_k >= C:
            # If we ask for >= number of channels just return the input
            # Note: in soft mode we would still return the original input (no scaling) for now to preserve
            # a consistent return shape (B, C, H, W) in this degenerate case.
            return x

        # Compute per-sample per-channel score: maximum absolute activation over spatial dims
        # Shape: (B, C)
        scores = x.abs().view(B, C, -1).max(dim=2).values

        if self.differentiable:
            # Differentiable relaxed top-K: we compute per-head logits over channels using
            # element-wise product between per-channel scores and a learned per-head parameter
            # then softmax across channels to create continuous, differentiable weights.
            # weights_param shape: (K, C)
            # scores shape: (B, C) -> expand to (B, K, C) for elementwise multiplication
            logits = scores.unsqueeze(1) * self.weights_param.unsqueeze(0)
            logits = logits / (self.temperature + self.const)
            weights = torch.softmax(logits, dim=-1)  # (B, K, C)

            # Combine channels via weighted sum -> output (B, K, H, W)
            # Expand dims: weights (B,K,C,1,1), x (B,1,C,H,W)
            out = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * x.unsqueeze(1), dim=2)
            return out

        # Get top-K indices per sample (descending order of score)
        _, topk_indices = torch.topk(scores, self.top_k, dim=1)

        # Expand indices to select spatially
        expanded_topk_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Use gather to select channels independently per sample
        selected_feature_maps = torch.gather(x, dim=1, index=expanded_topk_indices)

        if self.soft:
            # Compute softmax probabilities for channel scores (differentiable)
            probs = torch.softmax(scores / (self.temperature + self.const), dim=1)

            # Gather the probabilities corresponding to the selected top-K indices
            selected_probs = torch.gather(probs, dim=1, index=topk_indices)

            if self.normalize_weights:
                denom = selected_probs.sum(dim=1, keepdim=True).clamp_min(self.const)
                selected_probs = selected_probs / denom

            # Expand weights to spatial dims and scale selected channels
            selected_feature_maps = selected_feature_maps * selected_probs.unsqueeze(-1).unsqueeze(-1)

        return selected_feature_maps
