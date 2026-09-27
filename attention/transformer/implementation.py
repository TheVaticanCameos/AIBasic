"""Pre-LN Transformer Block；mask 由调用方明确传入。"""

import torch
from torch import Tensor, nn

from attention.feed_forward.implementation import FeedForward
from attention.multi_head.implementation import MultiHeadAttention
from layers.layer_norm.implementation import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1, self.norm2 = LayerNorm(dim), LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x [B,T,D] -> [B,T,D]；mask bool [T,T] 或 [B,1,T,T]，True=可见。"""
        x = x + self.dropout(self.attention(self.norm1(x), mask=mask))
        return x + self.dropout(self.ffn(self.norm2(x)))


if __name__ == "__main__":
    print(TransformerBlock(8, 2, 16)(torch.randn(2, 3, 8)).shape)
