"""离散策略与状态价值网络。"""

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class CategoricalPolicy(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, observations: Tensor) -> Tensor:
        """observations [B,O] -> logits [B,A]，不提前 softmax。"""
        return self.network(observations)

    def distribution(self, observations: Tensor) -> Categorical:
        """obs [B,O] -> batch_shape=[B], event_shape=[] 的分布。"""
        return Categorical(logits=self(observations))


class ValueNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, observations: Tensor) -> Tensor:
        """observations [B,O] -> values [B]，仅 squeeze 最后一维。"""
        return self.network(observations).squeeze(-1)


if __name__ == "__main__":
    print(CategoricalPolicy(4, 2).distribution(torch.zeros(3, 4)).sample())
