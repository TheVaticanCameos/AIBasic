"""正弦位置编码：练习骨架，不属于可运行标准实现。"""

import math
import torch
from torch import Tensor

def sinusoidal_encoding(length: int, dim: int, offset: int=0, device: torch.device | None=None, dtype: torch.dtype=torch.float32) -> Tensor:
    """返回 [T,D]，与 token embedding [B,T,D] 广播相加。"""
    raise NotImplementedError('手写练习：请完成核心实现')
