"""逆序计算分段轨迹 return，显式支持截断处的 bootstrap。"""

import torch
from torch import Tensor


@torch.no_grad()
def discounted_returns(
    rewards: Tensor, terminated: Tensor, truncated: Tensor, next_values: Tensor, gamma: float = 0.99
) -> Tensor:
    """全部输入 [T,B]，flags 为 bool，其余 floating -> returns [T,B]。

    next_values[t] 是该 transition 的真实下一观测价值，不能用 reset 后观测。
    terminated 不 bootstrap；truncated bootstrap 但阻断跨 episode 递推。
    数组末端也用 next_values[-1] bootstrap；纯 Monte Carlo 传全零即可。
    """
    result = torch.empty_like(rewards)
    running = next_values[-1].clone()  # [B]
    for t in reversed(range(rewards.shape[0])):
        continuation = torch.where(truncated[t], next_values[t], running)  # [B]
        running = rewards[t] + gamma * (~terminated[t]) * continuation
        result[t] = running
    return result


if __name__ == "__main__":
    r = torch.tensor([[1.0], [2.0]])
    flags = torch.zeros_like(r, dtype=torch.bool)
    print(discounted_returns(r, flags, flags, torch.zeros_like(r), 0.5))
