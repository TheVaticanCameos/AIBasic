"""张量操作与形状变换：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor


def image_to_tokens(x: Tensor) -> Tensor:
    """x [B,C,H,W] -> [B,H*W,C]；按行优先展平空间位置。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    b, c, h, w = x.shape
    return x.reshape(b, c, -1).transpose(1, 2)

def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """x [B,T,D] -> [B,H,T,D/H]。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    b, t, d = x.shape
    if num_heads <= 0 or d % num_heads != 0:
        raise ValueError("D must be divisible by positive num_heads")
    return x.reshape(b, t, num_heads, d // num_heads).transpose(1, 2)

def merge_heads(x: Tensor) -> Tensor:
    """x [B,H,T,d] -> [B,T,H*d]；transpose 后使用 reshape。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, -1)

def select_action_values(q: Tensor, actions: Tensor) -> Tensor:
    """q floating [B,A]，actions int64 [B] -> 所选动作值 [B]。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    return q.gather(1, actions[:, None]).squeeze(1)

def pairwise_dot(x: Tensor, y: Tensor) -> Tensor:
    """x [B,N,D], y [B,M,D] -> [B,N,M]；演示 einsum。"""
    # raise NotImplementedError('手写练习：请完成核心实现')
    return torch.einsum("bnd,bmd->bnm", x, y)
