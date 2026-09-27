import torch
from torch import nn

from layers.group_norm.implementation import GroupNorm
from tests.helpers import compare_input_gradients


def test_matches_torch_and_independent_of_other_samples() -> None:
    actual, expected = GroupNorm(2, 4).double(), nn.GroupNorm(2, 4).double()
    x = torch.randn(3, 4, 2, 2, dtype=torch.double)
    compare_input_gradients(actual, expected, x)
    torch.testing.assert_close(actual(x)[:1], actual(x[:1]))
