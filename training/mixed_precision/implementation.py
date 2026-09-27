"""AMP 单步训练：autocast + GradScaler + unscale 后裁剪。"""

from collections.abc import Callable

import torch
from torch import Tensor, nn


def amp_train_step(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    scaler: torch.amp.GradScaler,
    enabled: bool = True,
    max_grad_norm: float | None = None,
) -> float:
    """x [B,...], y [B,...] -> loss；model/x/y 必须在同一设备。

    CPU 用 bfloat16 + disabled scaler；CUDA 用 float16 + enabled scaler。
    scaler 应在训练循环外创建并跨 step 复用，以保留动态缩放状态。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    dtype = torch.float16 if x.device.type == "cuda" else torch.bfloat16
    with torch.autocast(device_type=x.device.type, dtype=dtype, enabled=enabled):
        loss = loss_fn(model(x), y)  # scalar []
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach())


if __name__ == "__main__":
    model = nn.Linear(3, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    print(
        amp_train_step(
            model,
            torch.randn(4, 3),
            torch.randn(4, 1),
            torch.optim.SGD(model.parameters(), lr=0.1),
            nn.MSELoss(),
            scaler,
        )
    )
