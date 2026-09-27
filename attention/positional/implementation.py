"""Sinusoidal 绝对位置编码，支持奇数维度与位置偏移。"""

import math

import torch
from torch import Tensor


def sinusoidal_encoding(
    length: int,
    dim: int,
    offset: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """返回 [T,D]，与 token embedding [B,T,D] 广播相加。"""
    positions = torch.arange(offset, offset + length, device=device, dtype=dtype)[:, None]
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )  # [ceil(D/2)]
    angles = positions * frequencies  # [T,ceil(D/2)]
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, : dim // 2].cos()
    return encoding


if __name__ == "__main__":
    print(sinusoidal_encoding(3, 5))
