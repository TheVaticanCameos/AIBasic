"""常见张量变换；B=batch，T=序列长度，D=隐藏维，H=heads。"""

import torch
from torch import Tensor


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """x [B,T,D] -> [B,H,T,D/H]。"""
    b, t, d = x.shape
    if num_heads <= 0 or d % num_heads:
        raise ValueError("D must be divisible by positive num_heads")
    return x.reshape(b, t, num_heads, d // num_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """x [B,H,T,d] -> [B,T,H*d]；transpose 后使用 reshape。"""
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)


def select_action_values(q: Tensor, actions: Tensor) -> Tensor:
    """q floating [B,A]，actions int64 [B] -> 所选动作值 [B]。"""
    return q.gather(1, actions[:, None]).squeeze(1)  # [B,1] -> [B]


def pairwise_dot(x: Tensor, y: Tensor) -> Tensor:
    """x [B,N,D], y [B,M,D] -> [B,N,M]；演示 einsum。"""
    return torch.einsum("bnd,bmd->bnm", x, y)


def image_to_tokens(x: Tensor) -> Tensor:
    """x [B,C,H,W] -> [B,H*W,C]；按行优先将空间位置展平为 token。"""
    b, c, h, w = x.shape
    channels_last = x.permute(0, 2, 3, 1)  # [B,H,W,C]，通常非连续
    return channels_last.reshape(b, h * w, c)  # [B,T,C]，T=H*W


if __name__ == "__main__":
    print(split_heads(torch.randn(2, 3, 8), 2).shape)
