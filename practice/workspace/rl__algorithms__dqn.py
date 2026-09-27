"""DQN / Double DQN / Dueling：练习骨架，不属于可运行标准实现。"""

from copy import deepcopy
import torch
from torch import Tensor, nn
from fundamentals.tensor_ops.implementation import select_action_values
from losses.regression.implementation import huber
from rl.buffers.replay.implementation import TransitionBatch
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import soft_update

class QNetwork(nn.Module):

    def __init__(self, observation_dim: int, num_actions: int, hidden_dim: int=64, dueling: bool=False) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, observations: Tensor) -> Tensor:
        """obs [B,O] -> Q [B,A]；dueling 的 advantage 按动作维减均值。"""
        raise NotImplementedError('手写练习：请完成核心实现')

class DQN:

    def __init__(self, observation_dim: int, num_actions: int, lr: float=0.001, gamma: float=0.99, tau: float=0.01, double: bool=True, dueling: bool=False) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def act(self, observations: Tensor, epsilon: float=0.0) -> Tensor:
        """obs [B,O] -> int64 actions [B]；每个样本独立 epsilon-greedy。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    @torch.no_grad()
    def targets(self, batch: TransitionBatch) -> Tensor:
        """返回 detached [B]；Double DQN 用 online 选动作、target 估值。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def update(self, batch: TransitionBatch) -> float:
        """obs [B,O], action [B]；一次优化后软更新 target。"""
        raise NotImplementedError('手写练习：请完成核心实现')
