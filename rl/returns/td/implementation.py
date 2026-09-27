"""单步 TD target 与优势标准化。"""

import torch
from torch import Tensor


@torch.no_grad()
def td_target(
    rewards: Tensor, next_values: Tensor, terminated: Tensor, gamma: float = 0.99
) -> Tensor:
    """rewards/next_values [B]，terminated bool [B] -> detached target [B]。

    时间限制 truncated 不传入 terminated，仍需要 bootstrap。
    """
    if rewards.shape != next_values.shape or rewards.shape != terminated.shape:
        raise ValueError("TD inputs must have identical shapes")
    return rewards + gamma * (~terminated) * next_values


def normalize_advantages(advantages: Tensor, eps: float = 1e-8) -> Tensor:
    """advantages [...] -> 同形状；总体标准差，单样本也不会 NaN。"""
    return (advantages - advantages.mean()) / (advantages.std(correction=0) + eps)


if __name__ == "__main__":
    print(td_target(torch.ones(2), torch.full((2,), 10.0), torch.tensor([True, False])))
