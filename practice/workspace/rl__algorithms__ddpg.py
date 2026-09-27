"""DDPG：练习骨架，不属于可运行标准实现。"""

from copy import deepcopy
import torch
from torch import Tensor
from rl.buffers.replay.implementation import TransitionBatch
from rl.policies.continuous.implementation import ContinuousQ, DeterministicPolicy
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import frozen_parameters, soft_update

class DDPG:

    def __init__(self, observation_dim: int, action_dim: int, lr: float=0.001, gamma: float=0.99, tau: float=0.005) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def act(self, observations: Tensor, noise_std: float=0.0) -> Tensor:
        """obs [B,O] -> 标准化动作 [B,A]；评估时 noise_std=0。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        """obs [B,O], actions [B,A]；动作已经映射到 [-1,1] 空间。"""
        raise NotImplementedError('手写练习：请完成核心实现')
