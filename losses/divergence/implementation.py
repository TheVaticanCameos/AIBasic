"""KL(P||Q)：类别求和，batch 求均值。"""

import torch
from torch import Tensor


def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    """log_p/log_q [B,C]，为归一化后的有限 log-prob -> scalar []。

    若 P 是监督目标，调用方传 log_p.detach()；本函数保留两侧梯度。
    """
    if log_p.shape != log_q.shape:
        raise ValueError("log probabilities must have identical shapes")
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean()


if __name__ == "__main__":
    p = torch.randn(2, 3).log_softmax(-1)
    print(kl_divergence(p, p))
