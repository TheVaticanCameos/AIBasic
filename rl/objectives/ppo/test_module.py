import torch

from rl.objectives.ppo.implementation import clipped_policy_loss


def test_positive_negative_advantage_clipping_and_gradient_boundaries() -> None:
    new = torch.tensor([1.5, 0.5]).log().requires_grad_()
    old = torch.zeros(2, requires_grad=True)
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    loss, stats = clipped_policy_loss(new, old, advantages)
    torch.testing.assert_close(loss, torch.tensor(-0.2))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.zeros(2))
    assert old.grad is advantages.grad is None
    assert stats.clip_fraction == 1


def test_unfavorable_ratios_are_not_clipped_away() -> None:
    new = torch.tensor([0.5, 1.5]).log().requires_grad_()
    loss, _ = clipped_policy_loss(new, torch.zeros(2), torch.tensor([1.0, -1.0]))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.tensor([-0.25, 0.75]))
