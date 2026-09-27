"""双向 InfoNCE：batch 内同索引为正样本，其余为负样本。"""

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.classification.implementation import cross_entropy


def info_nce(x: Tensor, y: Tensor, temperature: float = 0.1) -> Tensor:
    """x/y [B,D] -> scalar []；要求 B>=2，temperature>0。"""
    if x.shape != y.shape or x.shape[0] < 2 or temperature <= 0:
        raise ValueError("require matching [B,D], B>=2 and positive temperature")
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)  # [B,D]
    logits = x @ y.T / temperature  # [B,B]
    targets = torch.arange(x.shape[0], device=x.device)  # int64 [B]
    return (cross_entropy(logits, targets) + cross_entropy(logits.T, targets)) / 2


if __name__ == "__main__":
    print(info_nce(torch.eye(4), torch.eye(4)))
