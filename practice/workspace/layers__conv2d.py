"""Conv2d 循环与 unfold 实现：练习骨架，不属于可运行标准实现。"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

def conv2d_loop(x: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int=1, padding: int=0) -> Tensor:
    """x [B,C,H,W], weight [O,C,K,K], bias [O] -> [B,O,Hout,Wout]。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def conv2d_unfold(x: Tensor, weight: Tensor, bias: Tensor | None=None, stride: int=1, padding: int=0) -> Tensor:
    """同循环版接口；unfold 仅负责提取窗口，卷积计算使用矩阵乘法。"""
    raise NotImplementedError('手写练习：请完成核心实现')

class Conv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor) -> Tensor:
        """x [B,C,H,W] -> [B,O,Hout,Wout]。"""
        raise NotImplementedError('手写练习：请完成核心实现')
