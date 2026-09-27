"""连续动作高斯策略，支持重参数采样与 tanh 密度修正。"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """observations [B,O] -> (mean, log_std)，各 [B,A]。"""
        hidden = self.network(observations)  # [B,H]
        return self.mean(hidden), self.log_std(hidden).clamp(-20, 2)

    def sample(
        self, observations: Tensor, squash: bool = True, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        """obs [B,O] -> actions [B,A], log_probs [B]；squash 时动作范围 [-1,1]。

        log_prob 对动作维求和。环境边界缩放交给外层；本模块密度定义在标准动作空间。
        """
        mean, log_std = self(observations)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()  # [B,A]
        log_prob = distribution.log_prob(raw)  # [B,A]
        if squash:
            # log(1-tanh(u)^2) 的稳定表达，避免动作接近 ±1 时 log(0)。
            correction = 2 * (math.log(2.0) - raw - F.softplus(-2 * raw))
            log_prob = log_prob - correction
            raw = raw.tanh()
        return raw, log_prob.sum(dim=-1)


if __name__ == "__main__":
    print(GaussianPolicy(3, 2).sample(torch.zeros(4, 3))[0].shape)
