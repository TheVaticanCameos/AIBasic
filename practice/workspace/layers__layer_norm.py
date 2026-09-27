"""LayerNorm：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-05) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [...,D] -> [...,D]；使用总体方差 correction=0。"""
        raise NotImplementedError('手写练习：请完成核心实现')
