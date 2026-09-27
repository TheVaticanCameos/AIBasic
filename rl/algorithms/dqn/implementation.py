"""DQN / Double DQN，附可独立替换的 Dueling Q 网络。"""

from copy import deepcopy

import torch
from torch import Tensor, nn

from fundamentals.tensor_ops.implementation import select_action_values
from losses.regression.implementation import huber
from rl.buffers.replay.implementation import TransitionBatch
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import soft_update


class QNetwork(nn.Module):
    def __init__(
        self, observation_dim: int, num_actions: int, hidden_dim: int = 64, dueling: bool = False
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU())
        self.advantage = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1) if dueling else None

    def forward(self, observations: Tensor) -> Tensor:
        """obs [B,O] -> Q [B,A]；dueling 的 advantage 按动作维减均值。"""
        hidden = self.features(observations)  # [B,H]
        advantage = self.advantage(hidden)  # [B,A]
        if self.value is None:
            return advantage
        return self.value(hidden) + advantage - advantage.mean(-1, keepdim=True)


class DQN:
    def __init__(
        self,
        observation_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        double: bool = True,
        dueling: bool = False,
    ) -> None:
        self.online = QNetwork(observation_dim, num_actions, dueling=dueling)
        self.target = deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma, self.tau, self.double = gamma, tau, double
        self.num_actions = num_actions

    @torch.no_grad()
    def act(self, observations: Tensor, epsilon: float = 0.0) -> Tensor:
        """obs [B,O] -> int64 actions [B]；每个样本独立 epsilon-greedy。"""
        greedy = self.online(observations).argmax(-1)
        random_actions = torch.randint(self.num_actions, greedy.shape, device=greedy.device)
        explore = torch.rand(greedy.shape, device=greedy.device) < epsilon
        return torch.where(explore, random_actions, greedy)

    @torch.no_grad()
    def targets(self, batch: TransitionBatch) -> Tensor:
        """返回 detached [B]；Double DQN 用 online 选动作、target 估值。"""
        next_q = self.target(batch.next_observations)  # [B,A]
        if self.double:
            actions = self.online(batch.next_observations).argmax(-1)  # [B]
            next_values = select_action_values(next_q, actions)
        else:
            next_values = next_q.max(-1).values
        return td_target(batch.rewards, next_values, batch.terminated, self.gamma)

    def update(self, batch: TransitionBatch) -> float:
        """obs [B,O], action [B]；一次优化后软更新 target。"""
        predictions = select_action_values(self.online(batch.observations), batch.actions)
        loss = huber(predictions, self.targets(batch))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        soft_update(self.target, self.online, self.tau)
        return float(loss.detach())


if __name__ == "__main__":
    print(DQN(4, 2).act(torch.zeros(3, 4)))
