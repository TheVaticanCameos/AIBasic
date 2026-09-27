"""n-step return：遇到 episode 边界或数据边界提前停止。"""

import torch
from torch import Tensor


@torch.no_grad()
def n_step_returns(
    rewards: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    n: int = 3,
    gamma: float = 0.99,
) -> Tensor:
    """输入均 [T,B]，flags bool -> [T,B]；教学版 O(T*n*B)。"""
    if n < 1:
        raise ValueError("n must be positive")
    output = torch.zeros_like(rewards)
    t_size, batch = rewards.shape
    for start in range(t_size):
        active = torch.ones(batch, dtype=torch.bool, device=rewards.device)  # [B]
        for offset in range(min(n, t_size - start)):
            t = start + offset
            output[start] += gamma**offset * rewards[t] * active
            boundary = terminated[t] | truncated[t]
            horizon = offset == n - 1 or t == t_size - 1
            bootstrap = active & ~terminated[t] & (boundary | horizon)
            output[start] += gamma ** (offset + 1) * next_values[t] * bootstrap
            active = active & ~boundary
    return output


if __name__ == "__main__":
    r = torch.ones(3, 1)
    f = torch.zeros_like(r, dtype=torch.bool)
    print(n_step_returns(r, r * 0, f, f, n=2, gamma=0.5))
