"""Linear 全连接层：练习骨架，不属于可运行标准实现。"""

import math
import torch
from torch import Tensor, nn

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        # raise NotImplementedError('手写练习：请完成核心实现')
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.uniform_(self.weight, -math.sqrt(3.0 / in_features), math.sqrt(3.0 / in_features))

    def forward(self, x: Tensor) -> Tensor:
        """x [...,I] -> [...,O]，权重仅作用在最后一维。"""
        # raise NotImplementedError('手写练习：请完成核心实现')
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


if __name__ == "__main__":
    print(Linear(4, 3)(torch.randn(2, 4)).shape)
