import pytest
import torch

from rl.buffers.rollout.implementation import RolloutBatch, RolloutBuffer


def test_snapshot_is_detached_and_finish_clears() -> None:
    buffer = RolloutBuffer()
    x, f = torch.ones(2, requires_grad=True), torch.zeros(2, dtype=torch.bool)
    step = RolloutBatch(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long), x, x, x, x, f, f)
    buffer.add(step)
    with torch.no_grad():
        x.add_(10)
    batch = buffer.finish()
    torch.testing.assert_close(batch.log_probs, torch.ones(1, 2))
    assert not batch.log_probs.requires_grad
    with pytest.raises(ValueError):
        buffer.finish()
