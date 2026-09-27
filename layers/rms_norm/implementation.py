"""RMSNorm：不减均值，以均方根缩放。"""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [D]
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """x [...,D] -> [...,D]。"""
        # 半精度输入用 float32 累加，double 输入保持 double 精度。
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scale = torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)  # [...,1]
        return (work * scale).to(x.dtype) * self.weight


if __name__ == "__main__":
    print(RMSNorm(4)(torch.randn(2, 4)))
