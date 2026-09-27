"""Replay Buffer：练习骨架，不属于可运行标准实现。"""

from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class TransitionBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor
    truncated: Tensor

class ReplayBuffer:

    def __init__(self, capacity: int, observation_dim: int, action_dim: int | None=None) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def __len__(self) -> int:
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def add(self, observation: Tensor, action: Tensor, reward: float, next_observation: Tensor, terminated: bool, truncated: bool) -> None:
        """单条数据：obs/next_obs [O]，action [] 或 [A]；立即复制以免外部复用覆盖。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def sample(self, batch_size: int, generator: torch.Generator | None=None) -> TransitionBatch:
        """有放回采样，返回 B=batch_size；generator 用于复现。"""
        raise NotImplementedError('手写练习：请完成核心实现')
