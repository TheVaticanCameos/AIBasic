"""On-policy rollout：保存采样时的 log-prob/value，完成后整批消费。"""

from dataclasses import dataclass, fields

import torch
from torch import Tensor


@dataclass
class RolloutBatch:
    observations: Tensor  # [T,B,O]
    actions: Tensor  # [T,B]，离散动作
    rewards: Tensor  # [T,B]
    values: Tensor  # [T,B]，旧 value
    next_values: Tensor  # [T,B]，reset 前下一观测的旧 value
    log_probs: Tensor  # [T,B]，旧策略 log-prob
    terminated: Tensor  # bool [T,B]
    truncated: Tensor  # bool [T,B]


class RolloutBuffer:
    def __init__(self) -> None:
        self.steps: list[RolloutBatch] = []

    def add(self, step: RolloutBatch) -> None:
        """单步字段省略 T 维，各 [B,...]；detach + clone 固定旧策略数据。"""
        copies = {field.name: getattr(step, field.name).detach().clone() for field in fields(step)}
        self.steps.append(RolloutBatch(**copies))

    def finish(self) -> RolloutBatch:
        """按时间堆叠为 [T,B,...] 并清空，防止下一轮误用旧 rollout。"""
        if not self.steps:
            raise ValueError("empty rollout")
        data = {
            field.name: torch.stack([getattr(s, field.name) for s in self.steps])
            for field in fields(RolloutBatch)
        }
        self.steps.clear()
        return RolloutBatch(**data)


if __name__ == "__main__":
    buffer = RolloutBuffer()
    zero, flag = torch.zeros(1), torch.zeros(1, dtype=torch.bool)
    buffer.add(RolloutBatch(torch.zeros(1, 4), zero.long(), zero, zero, zero, zero, flag, flag))
    print(buffer.finish().observations.shape)
