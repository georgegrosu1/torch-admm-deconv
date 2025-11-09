from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F


class ParallelUpsampleReduce(nn.Module):
	"""Upsample input then fuse parallel stride convolutions back to original size."""

	def __init__(
		self,
		in_channels: int,
		scale_factor: int,
		num_branches: int,
        branch_kernel_size: list[int] | int,
		branch_channels: Optional[int] = None,
		branch_bias: bool = True,
		final_bias: bool = True,
		activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
	) -> None:
		super().__init__()

		if isinstance(branch_kernel_size, int):
			branch_kernel_size = [branch_kernel_size] * num_branches
		elif len(branch_kernel_size) != num_branches:
			raise ValueError("branch_kernel_size must be an int or a list of length num_branches")

		if scale_factor < 1 or int(scale_factor) != scale_factor:
			raise ValueError("scale_factor must be a positive integer")
		if num_branches < 1:
			raise ValueError("num_branches must be >= 1")
		if any(k % 2 == 0 for k in branch_kernel_size):
			raise ValueError(f"branch_kernel_size must be odd to preserve alignment but got {branch_kernel_size}")

		branch_channels = branch_channels or in_channels
		padding = [k // 2 for k in branch_kernel_size]

		self.scale_factor = int(scale_factor)
		self.branches = nn.ModuleList(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=k,
                stride=self.scale_factor,
                padding=p,
                bias=branch_bias,
            )
            for k, p in zip(branch_kernel_size, padding)
		)
		self.final_conv = nn.Conv2d(
			branch_channels * num_branches,
			in_channels,
			kernel_size=1,
			bias=final_bias,
		)
		self.activation = activation

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		upsampled = F.interpolate(
			x,
			scale_factor=self.scale_factor,
			mode="bicubic",
			align_corners=True,
		)

		branch_outputs = [branch(upsampled) for branch in self.branches]
		fused = torch.cat(branch_outputs, dim=1)
		reduced = self.final_conv(fused)

		return self.activation(reduced) if self.activation else reduced
