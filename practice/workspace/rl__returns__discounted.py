"""Discounted Return：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

@torch.no_grad()
def discounted_returns(rewards: Tensor, terminated: Tensor, truncated: Tensor, next_values: Tensor, gamma: float=0.99) -> Tensor:
    """全部输入 [T,B]，flags 为 bool，其余 floating -> returns [T,B]。

next_values[t] 是该 transition 的真实下一观测价值，不能用 reset 后观测。
terminated 不 bootstrap；truncated bootstrap 但阻断跨 episode 递推。
数组末端也用 next_values[-1] bootstrap；纯 Monte Carlo 传全零即可。"""
    raise NotImplementedError('手写练习：请完成核心实现')
