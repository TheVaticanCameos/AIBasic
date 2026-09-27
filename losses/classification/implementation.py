"""从 logits 计算稳定分类损失，固定返回 batch mean。"""

import torch
from torch import Tensor

from fundamentals.stable_ops.implementation import log_softmax


def cross_entropy(logits: Tensor, targets: Tensor, label_smoothing: float = 0.0) -> Tensor:
    """logits [B,C], targets int64 [B] -> scalar []。

    smoothing 使用 (1-eps)*one_hot + eps/C，与 PyTorch 定义一致。
    """
    if not 0 <= label_smoothing <= 1:
        raise ValueError("label_smoothing must be in [0,1]")
    log_probs = log_softmax(logits, dim=-1)  # [B,C]
    nll = -log_probs.gather(-1, targets[:, None]).squeeze(-1)  # [B]
    uniform_loss = -log_probs.mean(-1)  # [B]
    return ((1 - label_smoothing) * nll + label_smoothing * uniform_loss).mean()


def binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """logits/targets 同形状 [...]，targets 为 [0,1] 浮点值 -> scalar []。"""
    # logaddexp 避免 exp(logit) 溢出，并在 logit=0 处保留正确光滑梯度。
    return (torch.logaddexp(torch.zeros_like(logits), logits) - logits * targets).mean()


if __name__ == "__main__":
    print(cross_entropy(torch.tensor([[1000.0, 1001.0]]), torch.tensor([1])))
