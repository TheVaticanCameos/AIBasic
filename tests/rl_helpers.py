import torch

from rl.buffers.replay.implementation import TransitionBatch


def continuous_batch() -> TransitionBatch:
    return TransitionBatch(
        torch.randn(8, 3),
        torch.rand(8, 2) * 2 - 1,
        torch.randn(8),
        torch.randn(8, 3),
        torch.tensor([True, False] * 4),
        torch.zeros(8, dtype=torch.bool),
    )
