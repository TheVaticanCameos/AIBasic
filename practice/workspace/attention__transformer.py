"""Pre-LN Transformer Block：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn
from attention.feed_forward.implementation import FeedForward
from attention.multi_head.implementation import MultiHeadAttention
from layers.layer_norm.implementation import LayerNorm

class TransformerBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout: float=0.0) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor, mask: Tensor | None=None) -> Tensor:
        """x [B,T,D] -> [B,T,D]；mask bool [T,T] 或 [B,1,T,T]，True=可见。"""
        raise NotImplementedError('手写练习：请完成核心实现')
