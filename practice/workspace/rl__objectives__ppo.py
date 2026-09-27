"""PPO Clipped Objective：练习骨架，不属于可运行标准实现。"""

from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class PPOStats:
    approx_kl: float
    clip_fraction: float

def clipped_policy_loss(new_log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, clip_ratio: float=0.2) -> tuple[Tensor, PPOStats]:
    """输入均 [B] -> (loss [], stats)；仅 new_log_probs 接收梯度。"""
    raise NotImplementedError('手写练习：请完成核心实现')
