import torch
from torch import nn

from layers.layer_norm.implementation import LayerNorm
from tests.helpers import compare_input_gradients


def test_matches_torch_and_constant_input() -> None:
    actual, expected = LayerNorm(4).double(), nn.LayerNorm(4).double()
    compare_input_gradients(actual, expected, torch.randn(2, 3, 4, dtype=torch.double))
    torch.testing.assert_close(actual.weight.grad, expected.weight.grad)
    torch.testing.assert_close(
        actual(torch.ones(1, 4, dtype=torch.double)), torch.zeros(1, 4, dtype=torch.double)
    )
