"""RMSNorm：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [...,D] -> [...,D]。"""
        raise NotImplementedError('手写练习：请完成核心实现')
