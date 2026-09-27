"""一步 Actor-Critic：TD target 训练 value，TD error 作为 policy 权重。"""

import torch

from losses.regression.implementation import mse
from rl.buffers.replay.implementation import TransitionBatch
from rl.objectives.policy_gradient.implementation import policy_gradient_loss
from rl.policies.categorical.implementation import CategoricalPolicy, ValueNetwork
from rl.returns.td.implementation import td_target


def actor_critic_update(
    policy: CategoricalPolicy,
    value: ValueNetwork,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    """batch 的 obs [B,O]，actions [B]；要求刚采集的 on-policy transition。"""
    with torch.no_grad():
        targets = td_target(batch.rewards, value(batch.next_observations), batch.terminated, gamma)
    predictions = value(batch.observations)  # [B]
    advantages = targets - predictions.detach()  # [B]
    distribution = policy.distribution(batch.observations)
    actor_loss = policy_gradient_loss(
        distribution.log_prob(batch.actions), advantages, distribution.entropy(), entropy_coef
    )
    critic_loss = mse(predictions, targets)
    policy_optimizer.zero_grad()
    actor_loss.backward()
    policy_optimizer.step()
    value_optimizer.zero_grad()
    critic_loss.backward()
    value_optimizer.step()
    return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}


if __name__ == "__main__":
    p, v = CategoricalPolicy(4, 2), ValueNetwork(4)
    b = TransitionBatch(
        torch.randn(3, 4),
        torch.zeros(3, dtype=torch.long),
        torch.ones(3),
        torch.randn(3, 4),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
    )
    print(
        actor_critic_update(
            p, v, torch.optim.Adam(p.parameters()), torch.optim.Adam(v.parameters()), b
        )
    )
