"""Scaled Dot-Product Attention：练习骨架，不属于可运行标准实现。"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor

def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None=None, dropout_p: float=0.0, training: bool=False) -> tuple[Tensor, Tensor]:
    """q [...,Tq,d], k [...,Tk,d], v [...,Tk,dv] -> output [...,Tq,dv], weights [...,Tq,Tk]。

mask: bool，可广播到 [...,Tq,Tk]，True=可见。全被遮挡行定义为零输出。
返回的 weights 是 dropout 之后的权重，训练时行和不保证等于 1。"""
    raise NotImplementedError('手写练习：请完成核心实现')

def causal_mask(query_length: int, key_length: int, offset: int=0, device: torch.device | None=None) -> Tensor:
    """返回 bool [Tq,Tk]；query 的绝对位置为 offset + arange(Tq)。"""
    raise NotImplementedError('手写练习：请完成核心实现')
