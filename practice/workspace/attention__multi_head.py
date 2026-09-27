"""MHA / GQA / MQA / Cross-Attention：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor, nn
from attention.scaled_dot_product.implementation import attention
from fundamentals.tensor_ops.implementation import merge_heads, split_heads

class MultiHeadAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int | None=None, dropout: float=0.0) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

    def forward(self, x: Tensor, context: Tensor | None=None, mask: Tensor | None=None) -> Tensor:
        """x [B,Tq,D], context [B,Tk,D] -> [B,Tq,D]。

mask bool [Tq,Tk] 或 [B,1,Tq,Tk]；padding mask 应转换为 [B,1,1,Tk]。"""
        raise NotImplementedError('手写练习：请完成核心实现')
