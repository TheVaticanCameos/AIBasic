"""显式实现 Attention。布尔 mask 的 True 表示允许关注。"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[Tensor, Tensor]:
    """q [...,Tq,d], k [...,Tk,d], v [...,Tk,dv] -> output [...,Tq,dv], weights [...,Tq,Tk]。

    mask: bool，可广播到 [...,Tq,Tk]，True=可见。全被遮挡行定义为零输出。
    返回的 weights 是 dropout 之后的权重，训练时行和不保证等于 1。
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # [...,Tq,Tk]
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("mask must be bool; True means allowed")
        scores = scores.masked_fill(~mask, -torch.inf)
        empty = ~mask.any(dim=-1, keepdim=True)  # 可广播到 [...,Tq,1]
        scores = scores.masked_fill(empty, 0)  # 避免 softmax(-inf,...,-inf) 的 NaN
    weights = scores.softmax(dim=-1)
    if mask is not None:
        weights = weights.masked_fill(~mask, 0)
    weights = F.dropout(weights, p=dropout_p, training=training)
    return weights @ v, weights


def causal_mask(
    query_length: int, key_length: int, offset: int = 0, device: torch.device | None = None
) -> Tensor:
    """返回 bool [Tq,Tk]；query 的绝对位置为 offset + arange(Tq)。"""
    queries = torch.arange(query_length, device=device) + offset  # [Tq]
    keys = torch.arange(key_length, device=device)  # [Tk]
    return keys[None, :] <= queries[:, None]  # [Tq,Tk]


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    print(attention(x, x, x, causal_mask(3, 3))[0].shape)
