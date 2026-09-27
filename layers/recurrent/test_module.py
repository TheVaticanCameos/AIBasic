import pytest
import torch
from torch import nn

from layers.recurrent.implementation import GRUCell, LSTMCell, RNNCell


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru"])
def test_cells_match_pytorch_outputs_and_gradients(kind: str) -> None:
    own_cls = {"rnn": RNNCell, "lstm": LSTMCell, "gru": GRUCell}[kind]
    ref_cls = {"rnn": nn.RNNCell, "lstm": nn.LSTMCell, "gru": nn.GRUCell}[kind]
    own, ref = own_cls(3, 4).double(), ref_cls(3, 4).double()
    with torch.no_grad():
        ref.weight_ih.copy_(own.input.weight)
        ref.weight_hh.copy_(own.hidden.weight)
        ref.bias_ih.copy_(own.input.bias)
        ref.bias_hh.copy_(own.hidden.bias)
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    h = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    c = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    actual = own(x, (h, c)) if kind == "lstm" else own(x, h)
    expected = ref(x, (h, c)) if kind == "lstm" else ref(x, h)
    if isinstance(actual, tuple):
        actual, expected = torch.cat(actual, -1), torch.cat(expected, -1)
    torch.testing.assert_close(actual, expected)
    variables = (x, h, c) if kind == "lstm" else (x, h)
    actual_grads = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), variables)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b)
