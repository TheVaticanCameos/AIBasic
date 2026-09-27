"""Gaussian Policy 与 Tanh 修正：练习骨架，不属于可运行标准实现。"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

class GaussianPolicy(nn.Module):

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int=64) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """observations [B,O] -> (mean, log_std)，各 [B,A]。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def sample(self, observations: Tensor, squash: bool=True, deterministic: bool=False) -> tuple[Tensor, Tensor]:
        """obs [B,O] -> actions [B,A], log_probs [B]；squash 时动作范围 [-1,1]。

log_prob 对动作维求和。环境边界缩放交给外层；本模块密度定义在标准动作空间。"""
        raise NotImplementedError('手写练习：请完成核心实现')
