"""Rollout Buffer：练习骨架，不属于可运行标准实现。"""

from dataclasses import dataclass, fields
import torch
from torch import Tensor

@dataclass
class RolloutBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    values: Tensor
    next_values: Tensor
    log_probs: Tensor
    terminated: Tensor
    truncated: Tensor

class RolloutBuffer:

    def __init__(self) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def add(self, step: RolloutBatch) -> None:
        """单步字段省略 T 维，各 [B,...]；detach + clone 固定旧策略数据。"""
        raise NotImplementedError('手写练习：请完成核心实现')

    def finish(self) -> RolloutBatch:
        """按时间堆叠为 [T,B,...] 并清空，防止下一轮误用旧 rollout。"""
        raise NotImplementedError('手写练习：请完成核心实现')
