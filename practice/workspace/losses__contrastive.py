"""双向 InfoNCE：练习骨架，不属于可运行标准实现。"""

import torch
import torch.nn.functional as F
from torch import Tensor
from losses.classification.implementation import cross_entropy

def info_nce(x: Tensor, y: Tensor, temperature: float=0.1) -> Tensor:
    """x/y [B,D] -> scalar []；要求 B>=2，temperature>0。"""
    raise NotImplementedError('手写练习：请完成核心实现')
