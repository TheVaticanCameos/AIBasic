"""RoPE 旋转位置编码：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

def apply_rope(x: Tensor, offset: int=0, base: float=10000.0) -> Tensor:
    """x [...,T,d] -> 同形状；d 必须为偶数，位置从 offset 开始。"""
    raise NotImplementedError('手写练习：请完成核心实现')
