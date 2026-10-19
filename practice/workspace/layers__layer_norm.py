"""LayerNorm：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-05) -> None:
        # raise NotImplementedError('手写练习：请完成核心实现')
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """x [...,D] -> [...,D]；使用总体方差 correction=0。"""
        # raise NotImplementedError('手写练习：请完成核心实现')
        mu = x.mean(dim=-1, keepdim=True)
        return (x - mu) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps) * self.weight + self.bias
