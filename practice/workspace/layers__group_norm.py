"""GroupNorm：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class GroupNorm(nn.Module):

    def __init__(self, groups: int, channels: int, eps: float=1e-05) -> None:
        # raise NotImplementedError('手写练习：请完成核心实现')
        super().__init__()
        if groups <= 0 or channels % groups != 0:
            raise ValueError("channels must be divisible py positive groups!")
        self.groups = groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,...] -> 同形状。"""
        # raise NotImplementedError('手写练习：请完成核心实现')
        grouped = x.reshape(x.shape[0], self.groups, -1)    # [B, G, -1]
        mean = grouped.mean(dim=-1, keepdim=True) # [B, G, 1]
        var = grouped.var(dim=-1, keepdim=True, unbiased=False)   # [B, G, 1]
        normalized = (grouped - mean) * torch.rsqrt(var + self.eps) # [B, G, -1]
        normalized = normalized.reshape(x.shape)    # [B, C, ...]
        shape = (1, x.shape[1], *([1] * (x.ndim - 2)))
        return normalized * self.weight.reshape(shape) + self.bias.reshape(shape)
