import torch
import torch.nn as nn

from admmtor.elayers.cwa import ChannelWiseAttention, ChannelCompression


class AttentionChannelPooling(nn.Module):
    def __init__(self,
                 in_channels: int,
                 select_channels: int,
                 compressions: list[ChannelCompression] = (
                         ChannelCompression.STD, ChannelCompression.MEDIAN, ChannelCompression.MAX),
                 probas_channels_factor: int = 2,
                 reduce_probas_space: bool = False):
        super(AttentionChannelPooling, self).__init__()

        self.select_channels = select_channels
        self.compressions = compressions
        self.probas_channels_factor = probas_channels_factor
        self.reduce_probas_space = reduce_probas_space

        self.cwa = ChannelWiseAttention(in_channels, compressions, probas_channels_factor,
                                        reduce_probas_space=reduce_probas_space, reduce_mean=True, probas_only=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        channels_probabilities = self.cwa(x)

        _, top_n_indices = torch.topk(channels_probabilities, self.select_channels, dim=-1)
        expanded_top_n_indices = top_n_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Use gather to select channels for each batch element independently
        selected_feature_maps = torch.gather(x, dim=1, index=expanded_top_n_indices)

        return selected_feature_maps