"""n-step Return：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

@torch.no_grad()
def n_step_returns(rewards: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor, n: int=3, gamma: float=0.99) -> Tensor:
    """输入均 [T,B]，flags bool -> [T,B]；教学版 O(T*n*B)。"""
    raise NotImplementedError('手写练习：请完成核心实现')
