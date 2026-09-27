import torch
import torch.nn.functional as F

from losses.classification.implementation import binary_cross_entropy_with_logits, cross_entropy
from tests.helpers import compare_input_gradients


def test_cross_entropy_smoothing_and_extreme_logits() -> None:
    x = torch.tensor([[1000.0, -1000.0, 0.0], [1.0, 2.0, 3.0]], dtype=torch.double)
    target = torch.tensor([1, 2])
    for smoothing in (0.0, 0.2, 1.0):
        compare_input_gradients(
            lambda t: cross_entropy(t, target, smoothing),
            lambda t: F.cross_entropy(t, target, label_smoothing=smoothing),
            x,
        )


def test_bce_values_and_gradient_at_zero() -> None:
    x = torch.tensor([-1000.0, 0.0, 1000.0], dtype=torch.double)
    target = torch.tensor([1.0, 0.3, 0.0], dtype=torch.double)
    compare_input_gradients(
        lambda t: binary_cross_entropy_with_logits(t, target),
        lambda t: F.binary_cross_entropy_with_logits(t, target),
        x,
    )
