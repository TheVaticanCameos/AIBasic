"""PPO 算法组装：练习骨架，不属于可运行标准实现。"""

from dataclasses import dataclass
import torch
from torch import Tensor
from rl.buffers.rollout.implementation import RolloutBatch
from rl.objectives.ppo.implementation import clipped_policy_loss
from rl.policies.categorical.implementation import CategoricalPolicy, ValueNetwork
from rl.returns.gae.implementation import generalized_advantage_estimate
from rl.returns.td.implementation import normalize_advantages

@dataclass
class PPOConfig:
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5

class PPO:

    def __init__(self, observation_dim: int, num_actions: int, config: PPOConfig | None=None) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def act(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """obs [B,O] -> action/log_prob/value，各 [B]；固定采样时策略统计量。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        """rollout [T,B,...] -> 标量日志；展平为 N=T*B，旧数据始终 detach。"""
        raise NotImplementedError('手写练习：请完成核心实现')
