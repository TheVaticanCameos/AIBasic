"""一步 Actor-Critic：练习骨架，不属于可运行标准实现。"""

import torch
from losses.regression.implementation import mse
from rl.buffers.replay.implementation import TransitionBatch
from rl.objectives.policy_gradient.implementation import policy_gradient_loss
from rl.policies.categorical.implementation import CategoricalPolicy, ValueNetwork
from rl.returns.td.implementation import td_target

def actor_critic_update(policy: CategoricalPolicy, value: ValueNetwork, policy_optimizer: torch.optim.Optimizer, value_optimizer: torch.optim.Optimizer, batch: TransitionBatch, gamma: float=0.99, entropy_coef: float=0.01) -> dict[str, float]:
    """batch 的 obs [B,O]，actions [B]；要求刚采集的 on-policy transition。"""
    raise NotImplementedError('手写练习：请完成核心实现')
