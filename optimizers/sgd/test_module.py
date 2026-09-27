import torch

from optimizers.sgd.implementation import SGD


def test_multiple_steps_match_torch_with_decay_and_skipped_gradient() -> None:
    p = torch.randn(3, dtype=torch.double, requires_grad=True)
    q = p.detach().clone().requires_grad_()
    own = SGD([p], lr=0.03, momentum=0.9, weight_decay=0.1)
    ref = torch.optim.SGD([q], lr=0.03, momentum=0.9, weight_decay=0.1)
    for step in range(5):
        gradient = None if step == 2 else torch.randn_like(p)
        p.grad = gradient
        q.grad = None if gradient is None else gradient.clone()
        own.step()
        ref.step()
        torch.testing.assert_close(p, q)
    own.zero_grad()
    assert p.grad is None
