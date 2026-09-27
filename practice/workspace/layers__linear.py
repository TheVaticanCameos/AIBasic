"""Linear 全连接层：练习骨架，不属于可运行标准实现。"""

import math
import torch
from torch import Tensor, nn

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [...,I] -> [...,O]，权重仅作用在最后一维。"""
        raise NotImplementedError('手写练习：请完成核心实现')
