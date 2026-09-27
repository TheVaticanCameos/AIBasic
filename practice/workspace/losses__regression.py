"""MSE 与 Huber：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

def mse(prediction: Tensor, target: Tensor) -> Tensor:
    """prediction/target 同形状 [...] -> scalar []；所有元素取均值。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def huber(prediction: Tensor, target: Tensor, delta: float=1.0) -> Tensor:
    """同形状 [...] -> scalar []；delta>0，区别于缩放后的 SmoothL1。"""
    raise NotImplementedError('手写练习：请完成核心实现')
