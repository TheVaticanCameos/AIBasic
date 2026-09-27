"""Soft Actor-Critic：练习骨架，不属于可运行标准实现。"""

import math
from copy import deepcopy
import torch
from torch import Tensor, nn
from rl.buffers.replay.implementation import TransitionBatch
from rl.policies.continuous.implementation import ContinuousQ
from rl.policies.gaussian.implementation import GaussianPolicy
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import frozen_parameters, soft_update

class SAC:

    def __init__(self, observation_dim: int, action_dim: int, lr: float=0.0003, gamma: float=0.99, tau: float=0.005, alpha: float=0.2, automatic_temperature: bool=True) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    @property
    def alpha(self) -> Tensor:
        """温度 scalar []；策略与 critic 优化不应更新 alpha。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def act(self, observations: Tensor, deterministic: bool=False) -> Tensor:
        """obs [B,O] -> 标准动作 [B,A]，范围 [-1,1]。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def targets(self, batch: TransitionBatch) -> Tensor:
        """返回 entropy-regularized TD target [B]。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        """obs [B,O], actions [B,A]；依次更新 critic、actor、温度和 target。"""
        raise NotImplementedError('手写练习：请完成核心实现')
