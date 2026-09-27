"""Categorical Policy 与 Value Network：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

class CategoricalPolicy(nn.Module):

    def __init__(self, observation_dim: int, num_actions: int, hidden_dim: int=64) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, observations: Tensor) -> Tensor:
        """observations [B,O] -> logits [B,A]，不提前 softmax。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def distribution(self, observations: Tensor) -> Categorical:
        """obs [B,O] -> batch_shape=[B], event_shape=[] 的分布。"""
        raise NotImplementedError('手写练习：请完成核心实现')

class ValueNetwork(nn.Module):

    def __init__(self, observation_dim: int, hidden_dim: int=64) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, observations: Tensor) -> Tensor:
        """observations [B,O] -> values [B]，仅 squeeze 最后一维。"""
        raise NotImplementedError('手写练习：请完成核心实现')
