import torch
import torch.nn.functional as F

from layers.pooling.implementation import pool2d
from tests.helpers import compare_input_gradients


def test_pooling_output_and_gradients() -> None:
    x = torch.randn(2, 3, 7, 8, dtype=torch.double)
    compare_input_gradients(lambda t: pool2d(t, 3, 2, "max"), lambda t: F.max_pool2d(t, 3, 2), x)
    compare_input_gradients(lambda t: pool2d(t, 3, 2, "avg"), lambda t: F.avg_pool2d(t, 3, 2), x)
