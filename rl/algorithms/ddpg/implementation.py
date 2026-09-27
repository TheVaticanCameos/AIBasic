"""DDPG：确定性 actor、单 critic、目标 actor/critic、Polyak 更新。"""

from copy import deepcopy

import torch
from torch import Tensor

from rl.buffers.replay.implementation import TransitionBatch
from rl.policies.continuous.implementation import ContinuousQ, DeterministicPolicy
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import frozen_parameters, soft_update


class DDPG:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
    ) -> None:
        self.actor = DeterministicPolicy(observation_dim, action_dim)
        self.critic = ContinuousQ(observation_dim, action_dim)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau

    @torch.no_grad()
    def act(self, observations: Tensor, noise_std: float = 0.0) -> Tensor:
        """obs [B,O] -> 标准化动作 [B,A]；评估时 noise_std=0。"""
        actions = self.actor(observations)
        return (actions + noise_std * torch.randn_like(actions)).clamp(-1, 1)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        """obs [B,O], actions [B,A]；动作已经映射到 [-1,1] 空间。"""
        with torch.no_grad():
            next_actions = self.target_actor(batch.next_observations)  # [B,A]
            next_q = self.target_critic(batch.next_observations, next_actions)  # [B]
            targets = td_target(batch.rewards, next_q, batch.terminated, self.gamma)
        critic_loss = (self.critic(batch.observations, batch.actions) - targets).square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen_parameters(self.critic):
            actor_loss = -self.critic(batch.observations, self.actor(batch.observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
        }


if __name__ == "__main__":
    print(DDPG(3, 1).act(torch.zeros(2, 3)))
