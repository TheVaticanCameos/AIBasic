"""无 padding 的二维池化；支持方形窗口，floor 输出尺寸。"""

from typing import Literal

import torch
from torch import Tensor


def pool2d(
    x: Tensor, kernel_size: int = 2, stride: int | None = None, mode: Literal["max", "avg"] = "max"
) -> Tensor:
    """x [B,C,H,W] -> [B,C,Hout,Wout]。"""
    step = kernel_size if stride is None else stride
    windows = x.unfold(2, kernel_size, step).unfold(3, kernel_size, step)
    # windows: [B,C,Hout,Wout,K,K]
    if mode == "max":
        return windows.flatten(-2).max(-1).values  # [B,C,Hout,Wout]
    if mode == "avg":
        return windows.mean((-1, -2))
    raise ValueError("mode must be max or avg")


if __name__ == "__main__":
    print(pool2d(torch.arange(16.0).reshape(1, 1, 4, 4)))
