"""张量操作与形状变换：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor


def image_to_tokens(x: Tensor) -> Tensor:
    """x [B,C,H,W] -> [B,H*W,C]；按行优先展平空间位置。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """x [B,T,D] -> [B,H,T,D/H]。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def merge_heads(x: Tensor) -> Tensor:
    """x [B,H,T,d] -> [B,T,H*d]；transpose 后使用 reshape。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def select_action_values(q: Tensor, actions: Tensor) -> Tensor:
    """q floating [B,A]，actions int64 [B] -> 所选动作值 [B]。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def pairwise_dot(x: Tensor, y: Tensor) -> Tensor:
    """x [B,N,D], y [B,M,D] -> [B,N,M]；演示 einsum。"""
    raise NotImplementedError('手写练习：请完成核心实现')
