"""全连接层：y = x W^T + b。"""

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # [O,I]
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None  # [O]
        nn.init.uniform_(self.weight, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))

    def forward(self, x: Tensor) -> Tensor:
        """x [...,I] -> [...,O]，权重仅作用在最后一维。"""
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


if __name__ == "__main__":
    print(Linear(4, 3)(torch.randn(2, 4)).shape)
