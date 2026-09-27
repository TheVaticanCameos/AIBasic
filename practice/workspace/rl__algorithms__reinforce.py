"""REINFORCE 更新：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor
from rl.objectives.policy_gradient.implementation import policy_gradient_loss
from rl.policies.categorical.implementation import CategoricalPolicy
from rl.returns.td.implementation import normalize_advantages

def reinforce_update(policy: CategoricalPolicy, optimizer: torch.optim.Optimizer, observations: Tensor, actions: Tensor, returns: Tensor, normalize: bool=True) -> float:
    """obs [N,O], actions int64 [N], returns [N] -> loss；N 为轨迹总步数。

标准化是可选 variance-reduction 启发式；小 batch 下可能抹去学习信号。"""
    raise NotImplementedError('手写练习：请完成核心实现')
