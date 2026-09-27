"""连续控制的确定性 actor 与 Q(s,a)，供 DDPG/TD3/SAC 组合使用。"""

import torch
from torch import Tensor, nn


class DeterministicPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        """observations [B,O] -> actions [B,A]，范围 [-1,1]。"""
        return self.network(observations)


class ContinuousQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        """obs [B,O], actions [B,A] -> Q [B]。"""
        inputs = torch.cat((observations, actions), dim=-1)  # [B,O+A]
        return self.network(inputs).squeeze(-1)


if __name__ == "__main__":
    print(ContinuousQ(3, 2)(torch.zeros(4, 3), torch.zeros(4, 2)).shape)
