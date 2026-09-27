import pytest
import torch
from torch import nn

from layers.batch_norm.implementation import BatchNorm
from tests.helpers import compare_input_gradients


def test_train_eval_running_statistics_and_gradients() -> None:
    actual, expected = BatchNorm(3).double(), nn.BatchNorm2d(3).double()
    x = torch.randn(4, 3, 5, 5, dtype=torch.double)
    compare_input_gradients(actual, expected, x)
    torch.testing.assert_close(actual.weight.grad, expected.weight.grad)
    torch.testing.assert_close(actual.running_mean, expected.running_mean)
    torch.testing.assert_close(actual.running_var, expected.running_var)
    torch.testing.assert_close(actual.eval()(x[:1]), expected.eval()(x[:1]))
    with pytest.raises(ValueError):
        actual.train()(torch.ones(1, 3, dtype=torch.double))
