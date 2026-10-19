"""Inverted Dropout：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class Dropout(nn.Module):

    def __init__(self, p: float=0.5) -> None:
        # raise NotImplementedError('手写练习：请完成核心实现')
        super().__init__()
        if p > 1 or p < 0:
            raise ValueError("p must be in [0, 1]!")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """x [...] -> 同形状；每个元素独立保留。"""
        # raise NotImplementedError('手写练习：请完成核心实现')
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return x * 0
        
        keep = torch.rand_like(x) >= self.p
        return x * keep / (1 - self.p)
