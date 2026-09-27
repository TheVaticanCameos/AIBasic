"""一个逻辑 batch 的训练步骤，支持不等长 microbatch 的梯度累积。"""

from collections.abc import Callable, Iterable
from typing import Protocol

import torch
from torch import Tensor, nn


class OptimizerLike(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> object: ...


def clip_grad_norm(parameters: Iterable[Tensor], max_norm: float) -> Tensor:
    """梯度各为参数形状 [...]；返回裁剪前全局 L2 norm scalar []。"""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    norm = torch.stack([g.norm(2) for g in gradients]).norm(2)
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(scale)
    return norm


def train_step(
    model: nn.Module,
    microbatches: list[tuple[Tensor, Tensor]],
    optimizer: OptimizerLike,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    max_grad_norm: float | None = None,
) -> float:
    """每批 x [Bi,...], y [Bi,...]；loss_fn 返回样本平均损失 []。

    各批按 Bi/总样本数加权。要求 loss 可按样本分解；BatchNorm、Dropout 等
    会令 microbatch 与大 batch 的行为不同，不能据此宣称完全等价。
    """
    total = sum(x.shape[0] for x, _ in microbatches)
    if total == 0:
        raise ValueError("microbatches must contain samples")
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for x, y in microbatches:
        loss = loss_fn(model(x), y) * (x.shape[0] / total)  # scalar []
        loss.backward()
        total_loss += float(loss.detach())
    if max_grad_norm is not None:
        clip_grad_norm(model.parameters(), max_grad_norm)
    optimizer.step()
    return total_loss


if __name__ == "__main__":
    model = nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    print(train_step(model, [(torch.randn(4, 3), torch.randn(4, 1))], opt, nn.MSELoss()))
