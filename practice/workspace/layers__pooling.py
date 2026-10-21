"""Max / Average Pooling：练习骨架，不属于可运行标准实现。"""

from typing import Literal
import torch
from torch import Tensor

def pool2d(x: Tensor, kernel_size: int=2, stride: int | None=None, mode: Literal['max', 'avg']='max') -> Tensor:
    """x [B,C,H,W] -> [B,C,Hout,Wout]。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    step = kernel_size if stride is None else stride
    windows = x.unfold(2, kernel_size, step).unfold(3, kernel_size, step)
    if mode == "max":
        return windows.flatten(-2).max(-1).values
    if mode == "avg":
        return windows.mean((-1, -2))
    raise ValueError("mode must be max or avg")
