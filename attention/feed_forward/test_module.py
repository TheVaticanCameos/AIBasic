import torch

from attention.feed_forward.implementation import FeedForward


def test_tokens_independent_and_swiglu_all_projections_trainable() -> None:
    for swiglu in (False, True):
        model = FeedForward(4, 7, swiglu)
        x = torch.randn(2, 3, 4, requires_grad=True)
        torch.testing.assert_close(model(x)[:, 0], model(x[:, :1])[:, 0])
        model(x).square().sum().backward()
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
