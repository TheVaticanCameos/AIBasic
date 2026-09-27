import torch

from rl.returns.discounted.implementation import discounted_returns


def test_termination_truncation_and_rollout_bootstrap() -> None:
    rewards = torch.tensor([[1.0], [2.0], [100.0]])
    terminated = torch.tensor([[False], [False], [True]])
    truncated = torch.tensor([[False], [True], [False]])
    next_values = torch.tensor([[20.0], [30.0], [40.0]], requires_grad=True)
    result = discounted_returns(rewards, terminated, truncated, next_values, 0.5)
    torch.testing.assert_close(result, torch.tensor([[9.5], [17.0], [100.0]]))
    assert not result.requires_grad
    flags = torch.zeros(1, 1, dtype=torch.bool)
    torch.testing.assert_close(
        discounted_returns(rewards[:1], flags, flags, next_values[:1], 0.5), torch.tensor([[11.0]])
    )
