import torch
import torch.nn.functional as F

from layers.rms_norm.implementation import RMSNorm
from tests.helpers import compare_input_gradients


def test_rms_formula_and_gradient() -> None:
    model = RMSNorm(4, eps=1e-6).double()
    compare_input_gradients(
        model,
        lambda x: F.rms_norm(x, (4,), eps=1e-6),
        torch.randn(2, 3, 4, dtype=torch.double),
    )
    assert torch.isfinite(model(torch.zeros(2, 4, dtype=torch.double))).all()
