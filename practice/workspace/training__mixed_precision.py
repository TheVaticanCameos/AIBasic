"""AMP 混合精度训练：练习骨架，不属于可运行标准实现。"""

from collections.abc import Callable
import torch
from torch import Tensor, nn

def amp_train_step(model: nn.Module, x: Tensor, y: Tensor, optimizer: torch.optim.Optimizer, loss_fn: Callable[[Tensor, Tensor], Tensor], scaler: torch.amp.GradScaler, enabled: bool=True, max_grad_norm: float | None=None) -> float:
    """x [B,...], y [B,...] -> loss；model/x/y 必须在同一设备。

CPU 用 bfloat16 + disabled scaler；CUDA 用 float16 + enabled scaler。
scaler 应在训练循环外创建并跨 step 复用，以保留动态缩放状态。"""
    raise NotImplementedError('手写练习：请完成核心实现')
