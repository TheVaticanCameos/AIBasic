"""RoPE 相邻偶奇维配对版本；不同库可能采用前后半维配对约定。"""

import torch
from torch import Tensor


def apply_rope(x: Tensor, offset: int = 0, base: float = 10000.0) -> Tensor:
    """x [...,T,d] -> 同形状；d 必须为偶数，位置从 offset 开始。"""
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE requires an even head dimension")
    positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=x.dtype)
    inv_freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angles = positions[:, None] * inv_freq[None, :]  # [T,d/2]
    even, odd = x[..., 0::2], x[..., 1::2]  # [...,T,d/2]
    rotated = torch.stack(
        (even * angles.cos() - odd * angles.sin(), even * angles.sin() + odd * angles.cos()), dim=-1
    )
    return rotated.flatten(-2)  # [...,T,d/2,2] -> [...,T,d]


if __name__ == "__main__":
    print(apply_rope(torch.randn(2, 4, 3, 8)).shape)
