"""FFN 与 SwiGLU：练习骨架，不属于可运行标准实现。"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class FeedForward(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, swiglu: bool=False) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [B,T,D] -> [B,T,D]；hidden [B,T,F]。"""
        raise NotImplementedError('手写练习：请完成核心实现')
