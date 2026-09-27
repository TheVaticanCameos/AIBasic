import torch
from torch import nn

from layers.linear.implementation import Linear
from tests.helpers import compare_input_gradients


def test_output_input_and_parameter_gradients() -> None:
    actual, expected = Linear(4, 3).double(), nn.Linear(4, 3).double()
    expected.load_state_dict(actual.state_dict())
    compare_input_gradients(actual, expected, torch.randn(2, 5, 4, dtype=torch.double))
    torch.testing.assert_close(actual.weight.grad, expected.weight.grad)
    assert actual.bias is not None
    torch.testing.assert_close(actual.bias.grad, expected.bias.grad)
