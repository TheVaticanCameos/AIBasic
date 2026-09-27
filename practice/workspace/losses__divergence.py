"""KL Divergence：练习骨架，不属于可运行标准实现。"""

import torch
from torch import Tensor

def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    """log_p/log_q [B,C]，为归一化后的有限 log-prob -> scalar []。

若 P 是监督目标，调用方传 log_p.detach()；本函数保留两侧梯度。"""
    raise NotImplementedError('手写练习：请完成核心实现')
