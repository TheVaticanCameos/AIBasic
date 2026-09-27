"""Inverted Dropout：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class Dropout(nn.Module):

    def __init__(self, p: float=0.5) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [...] -> 同形状；每个元素独立保留。"""
        raise NotImplementedError('手写练习：请完成核心实现')
