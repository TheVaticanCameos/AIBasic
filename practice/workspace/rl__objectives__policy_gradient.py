"""Policy Gradient Loss：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

def policy_gradient_loss(log_probs: Tensor, weights: Tensor, entropy: Tensor | None=None, entropy_coef: float=0.0) -> Tensor:
    """log_probs/weights/entropy [B] -> scalar []；weights 明确停止梯度。"""
    raise NotImplementedError('手写练习：请完成核心实现')
