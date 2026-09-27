"""MHA / GQA / MQA；cross-attention 通过独立 context 输入表达。"""

import torch
from torch import Tensor, nn

from attention.scaled_dot_product.implementation import attention
from fundamentals.tensor_ops.implementation import merge_heads, split_heads


class MultiHeadAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, num_kv_heads: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads <= 0 or kv_heads <= 0 or dim % num_heads or num_heads % kv_heads:
            raise ValueError("require D % H == 0 and H % Hkv == 0")
        self.num_heads, self.num_kv_heads = num_heads, kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(
        self, x: Tensor, context: Tensor | None = None, mask: Tensor | None = None
    ) -> Tensor:
        """x [B,Tq,D], context [B,Tk,D] -> [B,Tq,D]。

        mask bool [Tq,Tk] 或 [B,1,Tq,Tk]；padding mask 应转换为 [B,1,1,Tk]。
        """
        context = x if context is None else context
        q = split_heads(self.q_proj(x), self.num_heads)  # [B,H,Tq,d]
        k = split_heads(self.k_proj(context), self.num_kv_heads)  # [B,Hkv,Tk,d]
        v = split_heads(self.v_proj(context), self.num_kv_heads)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)  # [B,H,Tk,d]，教学版显式展开
        v = v.repeat_interleave(repeats, dim=1)
        output, _ = attention(q, k, v, mask, self.dropout, self.training)
        return self.out_proj(merge_heads(output))  # [B,Tq,D]


if __name__ == "__main__":
    print(MultiHeadAttention(16, 4, 2)(torch.randn(2, 5, 16)).shape)
