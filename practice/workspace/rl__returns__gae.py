"""Generalized Advantage Estimation：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

@torch.no_grad()
def generalized_advantage_estimate(rewards: Tensor, values: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor, gamma: float=0.99, gae_lambda: float=0.95) -> tuple[Tensor, Tensor]:
    """所有输入 [T,B] -> (advantages, returns)，各 [T,B]，均无梯度。

next_values 使用 transition 的下一观测；episode reset 不得覆盖它。
δ=r+γ(1-terminated)V(next)-V；A=δ+γλ(1-ended)A_next。"""
    raise NotImplementedError('手写练习：请完成核心实现')
