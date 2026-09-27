"""TD3：练习骨架，不属于可运行标准实现。"""

from copy import deepcopy
import torch
from torch import Tensor
from rl.buffers.replay.implementation import TransitionBatch
from rl.policies.continuous.implementation import ContinuousQ, DeterministicPolicy
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import frozen_parameters, soft_update

class TD3:

    def __init__(self, observation_dim: int, action_dim: int, lr: float=0.001, gamma: float=0.99, tau: float=0.005, policy_delay: int=2, target_noise: float=0.2, noise_clip: float=0.5) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def act(self, observations: Tensor, noise_std: float=0.0) -> Tensor:
        """obs [B,O] -> actions [B,A]，范围 [-1,1]。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def targets(self, batch: TransitionBatch) -> Tensor:
        """返回 [B]；目标噪声与探索噪声相互独立。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def update(self, batch: TransitionBatch) -> dict[str, float | None]:
        """每次更新 critic；每 policy_delay 次更新 actor 与所有 targets。"""
        raise NotImplementedError('手写练习：请完成核心实现')
