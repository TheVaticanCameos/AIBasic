"""推理 KV Cache：练习骨架，不属于可运行标准实现。"""

from dataclasses import dataclass
import torch
from torch import Tensor
from attention.multi_head.implementation import MultiHeadAttention
from attention.scaled_dot_product.implementation import attention, causal_mask
from fundamentals.tensor_ops.implementation import merge_heads, split_heads

@dataclass
class KVCache:
    keys: Tensor | None = None
    values: Tensor | None = None

    def clear(self) -> None:
        raise NotImplementedError('手写练习：请完成核心实现')

@torch.no_grad()
def cached_attention(module: MultiHeadAttention, x: Tensor, cache: KVCache) -> Tensor:
    """x [B,Tnew,D] -> [B,Tnew,D]；按时间顺序调用，换序列必须 clear。

输入应已包含绝对位置编码。本函数不集成 RoPE 或 padding mask。"""
    raise NotImplementedError('手写练习：请完成核心实现')
