"""交叉熵、Label Smoothing、BCE with Logits：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor
from fundamentals.stable_ops.implementation import log_softmax

def cross_entropy(logits: Tensor, targets: Tensor, label_smoothing: float=0.0) -> Tensor:
    """logits [B,C], targets int64 [B] -> scalar []。

smoothing 使用 (1-eps)*one_hot + eps/C，与 PyTorch 定义一致。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """logits/targets 同形状 [...]，targets 为 [0,1] 浮点值 -> scalar []。"""
    raise NotImplementedError('手写练习：请完成核心实现')
