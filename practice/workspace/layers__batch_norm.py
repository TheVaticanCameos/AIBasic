"""BatchNorm：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class BatchNorm(nn.Module):
    running_mean: Tensor
    running_var: Tensor

    def __init__(self, channels: int, eps: float=1e-05, momentum: float=0.1) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,...] -> 同形状；每个 C 在 batch 与空间维度上归约。"""
        raise NotImplementedError('手写练习：请完成核心实现')
