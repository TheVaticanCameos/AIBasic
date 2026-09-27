"""现代 SAC：双 Q、重参数随机策略、可选自动温度，无单独 V 网络。"""

import math
from copy import deepcopy

import torch
from torch import Tensor, nn

from rl.buffers.replay.implementation import TransitionBatch
from rl.policies.continuous.implementation import ContinuousQ
from rl.policies.gaussian.implementation import GaussianPolicy
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import frozen_parameters, soft_update


class SAC:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_temperature: bool = True,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.actor = GaussianPolicy(observation_dim, action_dim)
        self.q1, self.q2 = (
            ContinuousQ(observation_dim, action_dim),
            ContinuousQ(observation_dim, action_dim),
        )
        self.target_q1, self.target_q2 = (
            deepcopy(self.q1).requires_grad_(False),
            deepcopy(self.q2).requires_grad_(False),
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))  # scalar []
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.automatic_temperature = automatic_temperature
        self.target_entropy = -float(action_dim)
        self.gamma, self.tau = gamma, tau

    @property
    def alpha(self) -> Tensor:
        """温度 scalar []；策略与 critic 优化不应更新 alpha。"""
        return self.log_alpha.detach().exp()

    @torch.no_grad()
    def act(self, observations: Tensor, deterministic: bool = False) -> Tensor:
        """obs [B,O] -> 标准动作 [B,A]，范围 [-1,1]。"""
        return self.actor.sample(observations, deterministic=deterministic)[0]

    @torch.no_grad()
    def targets(self, batch: TransitionBatch) -> Tensor:
        """返回 entropy-regularized TD target [B]。"""
        next_actions, next_log_probs = self.actor.sample(batch.next_observations)
        next_q = torch.minimum(
            self.target_q1(batch.next_observations, next_actions),
            self.target_q2(batch.next_observations, next_actions),
        )
        return td_target(
            batch.rewards, next_q - self.alpha * next_log_probs, batch.terminated, self.gamma
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        """obs [B,O], actions [B,A]；依次更新 critic、actor、温度和 target。"""
        targets = self.targets(batch)
        critic_loss = (self.q1(batch.observations, batch.actions) - targets).square().mean() + (
            self.q2(batch.observations, batch.actions) - targets
        ).square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.q1, self.q2):
            actions, log_probs = self.actor.sample(batch.observations)  # [B,A], [B]
            q = torch.minimum(
                self.q1(batch.observations, actions), self.q2(batch.observations, actions)
            )
            actor_loss = (self.alpha * log_probs - q).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        alpha_loss = torch.zeros(())
        if self.automatic_temperature:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        soft_update(self.target_q1, self.q1, self.tau)
        soft_update(self.target_q2, self.q2, self.tau)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha),
        }


if __name__ == "__main__":
    print(SAC(3, 1).act(torch.zeros(2, 3)))
