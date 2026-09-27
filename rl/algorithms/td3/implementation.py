"""TD3：双 critic 取最小值、目标策略平滑、延迟 actor/target 更新。"""

from copy import deepcopy

import torch
from torch import Tensor

from rl.buffers.replay.implementation import TransitionBatch
from rl.policies.continuous.implementation import ContinuousQ, DeterministicPolicy
from rl.returns.td.implementation import td_target
from rl.updates.target.implementation import frozen_parameters, soft_update


class TD3:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
    ) -> None:
        if policy_delay < 1:
            raise ValueError("policy_delay must be positive")
        self.actor = DeterministicPolicy(observation_dim, action_dim)
        self.q1, self.q2 = (
            ContinuousQ(observation_dim, action_dim),
            ContinuousQ(observation_dim, action_dim),
        )
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_q1, self.target_q2 = (
            deepcopy(self.q1).requires_grad_(False),
            deepcopy(self.q2).requires_grad_(False),
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.gamma, self.tau, self.policy_delay = gamma, tau, policy_delay
        self.target_noise, self.noise_clip = target_noise, noise_clip
        self.steps = 0

    @torch.no_grad()
    def act(self, observations: Tensor, noise_std: float = 0.0) -> Tensor:
        """obs [B,O] -> actions [B,A]，范围 [-1,1]。"""
        actions = self.actor(observations)
        return (actions + noise_std * torch.randn_like(actions)).clamp(-1, 1)

    @torch.no_grad()
    def targets(self, batch: TransitionBatch) -> Tensor:
        """返回 [B]；目标噪声与探索噪声相互独立。"""
        actions = self.target_actor(batch.next_observations)
        noise = (torch.randn_like(actions) * self.target_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        actions = (actions + noise).clamp(-1, 1)  # [B,A]
        q = torch.minimum(
            self.target_q1(batch.next_observations, actions),
            self.target_q2(batch.next_observations, actions),
        )  # [B]
        return td_target(batch.rewards, q, batch.terminated, self.gamma)

    def update(self, batch: TransitionBatch) -> dict[str, float | None]:
        """每次更新 critic；每 policy_delay 次更新 actor 与所有 targets。"""
        self.steps += 1
        targets = self.targets(batch)
        critic_loss = (self.q1(batch.observations, batch.actions) - targets).square().mean() + (
            self.q2(batch.observations, batch.actions) - targets
        ).square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        actor_metric: float | None = None
        if self.steps % self.policy_delay == 0:
            with frozen_parameters(self.q1):
                actor_loss = -self.q1(batch.observations, self.actor(batch.observations)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            actor_metric = float(actor_loss.detach())
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_q1, self.q1, self.tau)
            soft_update(self.target_q2, self.q2, self.tau)
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": actor_metric}


if __name__ == "__main__":
    print(TD3(3, 1).act(torch.zeros(2, 3)))
