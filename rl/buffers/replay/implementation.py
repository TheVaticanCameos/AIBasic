"""预分配 CPU replay ring buffer；sample 返回独立的随机 batch。"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TransitionBatch:
    observations: Tensor  # float [B,O]
    actions: Tensor  # int64 [B] 或 float [B,A]
    rewards: Tensor  # float [B]
    next_observations: Tensor  # float [B,O]，reset 前的真实下一观测
    terminated: Tensor  # bool [B]
    truncated: Tensor  # bool [B]


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.position, self.size = capacity, 0, 0
        self.observations = torch.empty(capacity, observation_dim)
        self.next_observations = torch.empty_like(self.observations)
        self.actions = (
            torch.empty(capacity, dtype=torch.long)
            if action_dim is None
            else torch.empty(capacity, action_dim)
        )
        self.rewards = torch.empty(capacity)
        self.terminated = torch.empty(capacity, dtype=torch.bool)
        self.truncated = torch.empty(capacity, dtype=torch.bool)

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: float,
        next_observation: Tensor,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """单条数据：obs/next_obs [O]，action [] 或 [A]；立即复制以免外部复用覆盖。"""
        i = self.position
        self.observations[i].copy_(observation)
        self.actions[i].copy_(action)
        self.rewards[i] = reward
        self.next_observations[i].copy_(next_observation)
        self.terminated[i], self.truncated[i] = terminated, truncated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> TransitionBatch:
        """有放回采样，返回 B=batch_size；generator 用于复现。"""
        if self.size == 0 or batch_size <= 0:
            raise ValueError("require a nonempty buffer and positive batch_size")
        idx = torch.randint(self.size, (batch_size,), generator=generator)  # [B]
        return TransitionBatch(
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.terminated[idx],
            self.truncated[idx],
        )


if __name__ == "__main__":
    buffer = ReplayBuffer(10, 4)
    buffer.add(torch.zeros(4), torch.tensor(1), 1.0, torch.ones(4), False, True)
    print(buffer.sample(2).observations.shape)
