import torch

from attention.rotary.implementation import apply_rope


def test_norm_relative_positions_and_cached_offsets() -> None:
    x, y = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 4, 8)
    torch.testing.assert_close(apply_rope(x).norm(dim=-1), x.norm(dim=-1))
    # 对 q/k 同时平移绝对位置，pairwise score 不变。
    score = apply_rope(x) @ apply_rope(y).transpose(-2, -1)
    shifted = apply_rope(x, 10) @ apply_rope(y, 10).transpose(-2, -1)
    torch.testing.assert_close(score, shifted, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(apply_rope(x)[..., 2:, :], apply_rope(x[..., 2:, :], 2))
