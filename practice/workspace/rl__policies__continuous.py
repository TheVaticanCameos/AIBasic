"""确定性策略与 Continuous Q：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class DeterministicPolicy(nn.Module):

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int=64) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, observations: Tensor) -> Tensor:
        """observations [B,O] -> actions [B,A]，范围 [-1,1]。"""
        raise NotImplementedError('手写练习：请完成核心实现')

class ContinuousQ(nn.Module):

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int=64) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        """obs [B,O], actions [B,A] -> Q [B]。"""
        raise NotImplementedError('手写练习：请完成核心实现')
