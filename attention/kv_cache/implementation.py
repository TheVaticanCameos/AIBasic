"""推理专用 KV Cache；复用 MHA 权重，用位置偏移构造因果 mask。"""

from dataclasses import dataclass

import torch
from torch import Tensor

from attention.multi_head.implementation import MultiHeadAttention
from attention.scaled_dot_product.implementation import attention, causal_mask
from fundamentals.tensor_ops.implementation import merge_heads, split_heads


@dataclass
class KVCache:
    keys: Tensor | None = None  # [B,Hkv,Tpast,d]
    values: Tensor | None = None  # [B,Hkv,Tpast,d]

    def clear(self) -> None:
        self.keys, self.values = None, None


@torch.no_grad()
def cached_attention(module: MultiHeadAttention, x: Tensor, cache: KVCache) -> Tensor:
    """x [B,Tnew,D] -> [B,Tnew,D]；按时间顺序调用，换序列必须 clear。

    输入应已包含绝对位置编码。本函数不集成 RoPE 或 padding mask。
    """
    if module.training:
        raise ValueError("KV cache is inference-only; call eval() first")
    q = split_heads(module.q_proj(x), module.num_heads)  # [B,H,Tnew,d]
    k = split_heads(module.k_proj(x), module.num_kv_heads)
    v = split_heads(module.v_proj(x), module.num_kv_heads)
    offset = 0 if cache.keys is None else cache.keys.shape[-2]
    cache.keys = k if cache.keys is None else torch.cat((cache.keys, k), dim=-2)
    cache.values = v if cache.values is None else torch.cat((cache.values, v), dim=-2)
    repeats = module.num_heads // module.num_kv_heads
    k_all = cache.keys.repeat_interleave(repeats, dim=1)  # [B,H,Tpast+Tnew,d]
    v_all = cache.values.repeat_interleave(repeats, dim=1)
    mask = causal_mask(x.shape[1], k_all.shape[-2], offset, x.device)  # [Tnew,Ttotal]
    out, _ = attention(q, k_all, v_all, mask)
    return module.out_proj(merge_heads(out))


if __name__ == "__main__":
    model, cache = MultiHeadAttention(8, 2).eval(), KVCache()
    print(cached_attention(model, torch.randn(1, 1, 8), cache).shape)
