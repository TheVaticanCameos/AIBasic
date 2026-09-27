"""GroupNorm：每个样本独立，将通道分组后归一化。"""

import torch
from torch import Tensor, nn


class GroupNorm(nn.Module):
    def __init__(self, groups: int, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if groups <= 0 or channels % groups:
            raise ValueError("channels must be divisible by positive groups")
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))  # [C]
        self.bias = nn.Parameter(torch.zeros(channels))  # [C]

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,...] -> 同形状。"""
        grouped = x.reshape(x.shape[0], self.groups, -1)  # [B,G,C/G * spatial]
        mean = grouped.mean(-1, keepdim=True)  # [B,G,1]
        var = grouped.var(-1, correction=0, keepdim=True)  # [B,G,1]
        normalized = ((grouped - mean) * torch.rsqrt(var + self.eps)).reshape_as(x)
        affine_shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
        return normalized * self.weight.reshape(affine_shape) + self.bias.reshape(affine_shape)


if __name__ == "__main__":
    print(GroupNorm(2, 4)(torch.randn(2, 4, 3, 3)).shape)
