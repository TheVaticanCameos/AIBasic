import pytest
import torch

from rl.buffers.replay.implementation import ReplayBuffer


def test_overwrite_copy_sampling_and_terminal_flags() -> None:
    buffer = ReplayBuffer(2, 3)
    with pytest.raises(ValueError):
        buffer.sample(1)
    for i in range(3):
        observation = torch.full((3,), float(i), requires_grad=True)
        buffer.add(observation, torch.tensor(i % 2), float(i), observation + 1, False, i == 2)
        with torch.no_grad():
            observation.fill_(-99)
    assert len(buffer) == 2
    batch = buffer.sample(100, torch.Generator().manual_seed(0))
    assert set(batch.rewards.tolist()) == {1.0, 2.0}
    torch.testing.assert_close(batch.next_observations, batch.observations + 1)
    assert not batch.observations.requires_grad
    assert torch.equal(batch.truncated, batch.rewards == 2)
    assert not batch.terminated.any()
