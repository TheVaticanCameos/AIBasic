"""对最后 D 维特征归一化；不维护 running statistics。"""

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [D]
        self.bias = nn.Parameter(torch.zeros(dim))  # [D]
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """x [...,D] -> [...,D]；使用总体方差 correction=0。"""
        mean = x.mean(dim=-1, keepdim=True)  # [...,1]
        var = (x - mean).square().mean(dim=-1, keepdim=True)  # [...,1]
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


if __name__ == "__main__":
    print(LayerNorm(4)(torch.randn(2, 3, 4)).shape)
