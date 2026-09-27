"""Policy gradient 的 loss 形式：最大化回报转换为最小化负目标。"""

import torch
from torch import Tensor


def policy_gradient_loss(
    log_probs: Tensor, weights: Tensor, entropy: Tensor | None = None, entropy_coef: float = 0.0
) -> Tensor:
    """log_probs/weights/entropy [B] -> scalar []；weights 明确停止梯度。"""
    if log_probs.shape != weights.shape:
        raise ValueError("log_probs and weights must have identical shapes")
    loss = -(log_probs * weights.detach()).mean()
    if entropy is not None:
        loss = loss - entropy_coef * entropy.mean()
    return loss


if __name__ == "__main__":
    print(policy_gradient_loss(torch.tensor([-0.3, -0.5]), torch.tensor([1.0, -1.0])))
