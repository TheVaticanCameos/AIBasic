import torch

from rl.objectives.policy_gradient.implementation import policy_gradient_loss


def test_loss_sign_and_detached_weights() -> None:
    log_probs = torch.tensor([-1.0, -2.0], requires_grad=True)
    weights = torch.tensor([2.0, -3.0], requires_grad=True)
    loss = policy_gradient_loss(log_probs, weights)
    torch.testing.assert_close(loss, torch.tensor(-2.0))
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([-1.0, 1.5]))
    assert weights.grad is None
