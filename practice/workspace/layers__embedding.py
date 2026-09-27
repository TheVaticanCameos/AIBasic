"""Embedding 查表：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn

class Embedding(nn.Module):

    def __init__(self, vocab_size: int, dim: int) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, indices: Tensor) -> Tensor:
        """indices int64 [...] -> floating [...,D]；不实现 padding_idx/sparse。"""
        raise NotImplementedError('手写练习：请完成核心实现')
