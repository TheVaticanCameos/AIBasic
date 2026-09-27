"""离散 PPO：固定旧数据，计算 GAE，多轮 minibatch 更新。"""

from dataclasses import dataclass

import torch
from torch import Tensor

from rl.buffers.rollout.implementation import RolloutBatch
from rl.objectives.ppo.implementation import clipped_policy_loss
from rl.policies.categorical.implementation import CategoricalPolicy, ValueNetwork
from rl.returns.gae.implementation import generalized_advantage_estimate
from rl.returns.td.implementation import normalize_advantages


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5


class PPO:
    def __init__(
        self, observation_dim: int, num_actions: int, config: PPOConfig | None = None
    ) -> None:
        self.config = PPOConfig() if config is None else config
        if self.config.epochs < 1 or self.config.minibatch_size < 1:
            raise ValueError("epochs and minibatch_size must be positive")
        self.policy = CategoricalPolicy(observation_dim, num_actions)
        self.value = ValueNetwork(observation_dim)
        self.parameters = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.lr)

    @torch.no_grad()
    def act(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """obs [B,O] -> action/log_prob/value，各 [B]；固定采样时策略统计量。"""
        distribution = self.policy.distribution(observations)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), self.value(observations)

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        """rollout [T,B,...] -> 标量日志；展平为 N=T*B，旧数据始终 detach。"""
        cfg = self.config
        advantages, returns = generalized_advantage_estimate(
            rollout.rewards,
            rollout.values,
            rollout.next_values,
            rollout.terminated,
            rollout.truncated,
            cfg.gamma,
            cfg.gae_lambda,
        )
        advantages = normalize_advantages(advantages).flatten()  # [N]
        returns = returns.flatten()  # [N]
        obs = rollout.observations.detach().flatten(0, 1)  # [N,O]
        actions = rollout.actions.flatten()  # [N]
        old_log_probs = rollout.log_probs.detach().flatten()  # [N]
        n = actions.numel()
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        count = 0
        for _ in range(cfg.epochs):
            order = torch.randperm(n, device=obs.device)
            for indices in order.split(cfg.minibatch_size):
                distribution = self.policy.distribution(obs[indices])
                policy_loss, stats = clipped_policy_loss(
                    distribution.log_prob(actions[indices]),
                    old_log_probs[indices],
                    advantages[indices],
                    cfg.clip_ratio,
                )
                value_loss = (self.value(obs[indices]) - returns[indices]).square().mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * distribution.entropy().mean()
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters, cfg.max_grad_norm)
                self.optimizer.step()
                weight = indices.numel()
                totals["policy_loss"] += float(policy_loss.detach()) * weight
                totals["value_loss"] += float(value_loss.detach()) * weight
                totals["approx_kl"] += stats.approx_kl * weight
                totals["clip_fraction"] += stats.clip_fraction * weight
                count += weight
        return {key: value / count for key, value in totals.items()}


if __name__ == "__main__":
    print(PPO(4, 2).act(torch.zeros(3, 4)))
