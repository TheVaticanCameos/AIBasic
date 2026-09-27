"""回归损失：拒绝隐式 broadcasting，防止 [B,1] 与 [B] 变成 [B,B]。"""

import torch
from torch import Tensor


def mse(prediction: Tensor, target: Tensor) -> Tensor:
    """prediction/target 同形状 [...] -> scalar []；所有元素取均值。"""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return (prediction - target).square().mean()


def huber(prediction: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """同形状 [...] -> scalar []；delta>0，区别于缩放后的 SmoothL1。"""
    if prediction.shape != target.shape or delta <= 0:
        raise ValueError("require identical shapes and delta > 0")
    error = (prediction - target).abs()
    return torch.where(error <= delta, 0.5 * error.square(), delta * (error - 0.5 * delta)).mean()


if __name__ == "__main__":
    print(mse(torch.tensor([1.0, 2.0]), torch.zeros(2)))
