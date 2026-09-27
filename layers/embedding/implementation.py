"""Embedding 是按索引选权重行；重复索引的梯度会累加。"""

import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim) * 0.02)  # [V,D]

    def forward(self, indices: Tensor) -> Tensor:
        """indices int64 [...] -> floating [...,D]；不实现 padding_idx/sparse。"""
        return self.weight[indices]


if __name__ == "__main__":
    print(Embedding(10, 4)(torch.tensor([[1, 2, 1]])).shape)
