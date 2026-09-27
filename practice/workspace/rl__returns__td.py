"""TD Target 与优势标准化：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

@torch.no_grad()
def td_target(rewards: Tensor, next_values: Tensor, terminated: Tensor, gamma: float=0.99) -> Tensor:
    """rewards/next_values [B]，terminated bool [B] -> detached target [B]。

时间限制 truncated 不传入 terminated，仍需要 bootstrap。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def normalize_advantages(advantages: Tensor, eps: float=1e-08) -> Tensor:
    """advantages [...] -> 同形状；总体标准差，单样本也不会 NaN。"""
    raise NotImplementedError('手写练习：请完成核心实现')
