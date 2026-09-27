"""PPO clipped surrogate objective；每个数据点保守地取较小收益。"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PPOStats:
    approx_kl: float
    clip_fraction: float


def clipped_policy_loss(
    new_log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, clip_ratio: float = 0.2
) -> tuple[Tensor, PPOStats]:
    """输入均 [B] -> (loss [], stats)；仅 new_log_probs 接收梯度。"""
    if new_log_probs.shape != old_log_probs.shape or advantages.shape != old_log_probs.shape:
        raise ValueError("PPO inputs must have identical shapes")
    if not 0 < clip_ratio < 1:
        raise ValueError("clip_ratio must be in (0,1)")
    log_ratio = new_log_probs - old_log_probs.detach()  # [B]
    ratio = log_ratio.exp()  # [B]
    advantages = advantages.detach()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    with torch.no_grad():
        approx_kl = float(((ratio - 1) - log_ratio).mean())
        fraction = float(((ratio - 1).abs() > clip_ratio).float().mean())
    return loss, PPOStats(approx_kl, fraction)


if __name__ == "__main__":
    print(clipped_policy_loss(torch.zeros(2), torch.zeros(2), torch.tensor([1.0, -1.0])))
