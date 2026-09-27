"""Max / Average Pooling：练习骨架，不属于可运行标准实现。"""

from typing import Literal
import torch
from torch import Tensor

def pool2d(x: Tensor, kernel_size: int=2, stride: int | None=None, mode: Literal['max', 'avg']='max') -> Tensor:
    """x [B,C,H,W] -> [B,C,Hout,Wout]。"""
    raise NotImplementedError('手写练习：请完成核心实现')
