import torch

from fundamentals.stable_ops.implementation import log_softmax, logsumexp, softmax
from tests.helpers import compare_input_gradients


def test_large_logits_and_gradients() -> None:
    x = torch.tensor([[1000.0, 1001.0, -1000.0], [-1000.0, -1001.0, -999.0]], dtype=torch.double)
    compare_input_gradients(log_softmax, lambda v: v.log_softmax(-1), x)
    compare_input_gradients(softmax, lambda v: v.softmax(-1), x)
    compare_input_gradients(logsumexp, lambda v: v.logsumexp(-1), x)
    torch.testing.assert_close(softmax(x).sum(-1), torch.ones(2, dtype=torch.double))


def test_large_common_offset_preserves_uniform_probabilities() -> None:
    x = torch.full((2, 4), 1e20)
    torch.testing.assert_close(log_softmax(x), x.log_softmax(-1))
    torch.testing.assert_close(softmax(x), torch.full_like(x, 0.25))
