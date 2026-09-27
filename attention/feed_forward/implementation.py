"""逐 token FFN 与 SwiGLU；不混合序列位置。"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, swiglu: bool = False) -> None:
        super().__init__()
        self.swiglu = swiglu
        self.up = nn.Linear(dim, hidden_dim * (2 if swiglu else 1))
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """x [B,T,D] -> [B,T,D]；hidden [B,T,F]。"""
        projected = self.up(x)
        if self.swiglu:
            gate, value = projected.chunk(2, dim=-1)  # 各 [B,T,F]
            hidden = F.silu(gate) * value
        else:
            hidden = F.gelu(projected)
        return self.down(hidden)


if __name__ == "__main__":
    print(FeedForward(8, 16, swiglu=True)(torch.randn(2, 3, 8)).shape)
