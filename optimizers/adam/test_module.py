import pytest
import torch

from optimizers.adam.implementation import Adam, AdamW


@pytest.mark.parametrize("decoupled", [False, True])
def test_adam_matches_torch_across_steps_and_missing_gradients(decoupled: bool) -> None:
    p = torch.randn(3, dtype=torch.double, requires_grad=True)
    q = p.detach().clone().requires_grad_()
    own_cls = AdamW if decoupled else Adam
    ref_cls = torch.optim.AdamW if decoupled else torch.optim.Adam
    own, ref = own_cls([p], lr=0.01, weight_decay=0.1), ref_cls([q], lr=0.01, weight_decay=0.1)
    for step in range(6):
        gradient = None if step == 2 else torch.randn_like(p)
        p.grad, q.grad = gradient, None if gradient is None else gradient.clone()
        own.step()
        ref.step()
        torch.testing.assert_close(p, q, atol=1e-10, rtol=1e-8)
    assert own.state[0].step == 5
