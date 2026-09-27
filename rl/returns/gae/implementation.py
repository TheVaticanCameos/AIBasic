"""GAE 的 bootstrap mask 与 trace mask 分开表达。"""

import torch
from torch import Tensor


@torch.no_grad()
def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """所有输入 [T,B] -> (advantages, returns)，各 [T,B]，均无梯度。

    next_values 使用 transition 的下一观测；episode reset 不得覆盖它。
    δ=r+γ(1-terminated)V(next)-V；A=δ+γλ(1-ended)A_next。
    """
    advantages = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[0])  # [B]，rollout 边界截断 trace
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
        trace_mask = ~(terminated[t] | truncated[t])  # bool [B]
        running = delta + gamma * gae_lambda * trace_mask * running
        advantages[t] = running
    return advantages, advantages + values


if __name__ == "__main__":
    r = torch.ones(3, 1)
    flags = torch.zeros_like(r, dtype=torch.bool)
    print(generalized_advantage_estimate(r, r * 0, r * 0, flags, flags)[0])
