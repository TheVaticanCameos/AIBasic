import pytest
import torch
import torch.nn.functional as F

from losses.regression.implementation import huber, mse
from tests.helpers import compare_input_gradients


def test_losses_match_and_reject_broadcasting() -> None:
    target, prediction = (
        torch.randn(4, 3, dtype=torch.double),
        torch.randn(4, 3, dtype=torch.double),
    )
    compare_input_gradients(lambda x: mse(x, target), lambda x: F.mse_loss(x, target), prediction)
    compare_input_gradients(
        lambda x: huber(x, target, 2), lambda x: F.huber_loss(x, target, delta=2), prediction
    )
    with pytest.raises(ValueError):
        mse(torch.ones(3, 1), torch.ones(3))
