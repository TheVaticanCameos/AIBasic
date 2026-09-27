"""Inverted dropout：训练阶段缩放，推理阶段恒等。"""

import torch
from torch import Tensor, nn


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0,1]")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """x [...] -> 同形状；每个元素独立保留。"""
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return x * 0  # 保留计算图，梯度为零
        keep = torch.rand_like(x) >= self.p  # bool [...]
        return x * keep / (1 - self.p)


if __name__ == "__main__":
    print(Dropout()(torch.ones(2, 4)))
