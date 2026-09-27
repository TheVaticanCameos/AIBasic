import torch
import torch.nn.functional as F

from attention.scaled_dot_product.implementation import attention, causal_mask


def test_sdpa_output_and_all_gradients() -> None:
    q = torch.randn(2, 3, 4, 5, dtype=torch.double, requires_grad=True)
    k = torch.randn(2, 3, 6, 5, dtype=torch.double, requires_grad=True)
    v = torch.randn(2, 3, 6, 7, dtype=torch.double, requires_grad=True)
    mask = torch.rand(2, 1, 4, 6) > 0.3
    actual, _ = attention(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def test_fully_masked_rows_have_finite_zero_outputs_and_gradients() -> None:
    q = torch.randn(1, 3, 4, requires_grad=True)
    result, weights = attention(q, q, q, torch.zeros(3, 3, dtype=torch.bool))
    assert torch.count_nonzero(result) == torch.count_nonzero(weights) == 0
    result.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_causal_attention_ignores_future() -> None:
    x = torch.randn(2, 4, 3)
    changed = x.clone()
    changed[:, 2:] += 100
    mask = causal_mask(4, 4)
    torch.testing.assert_close(
        attention(x, x, x, mask)[0][:, :2], attention(changed, changed, changed, mask)[0][:, :2]
    )
