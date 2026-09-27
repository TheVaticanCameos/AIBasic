import torch
import torch.nn.functional as F

from losses.divergence.implementation import kl_divergence
from tests.helpers import compare_input_gradients


def test_kl_direction_reduction_and_gradient() -> None:
    log_p = torch.randn(3, 4, dtype=torch.double).log_softmax(-1)
    log_q = torch.randn(3, 4, dtype=torch.double).log_softmax(-1)
    compare_input_gradients(
        lambda q: kl_divergence(log_p, q),
        lambda q: F.kl_div(q, log_p, reduction="batchmean", log_target=True),
        log_q,
    )
    torch.testing.assert_close(kl_divergence(log_p, log_p), torch.zeros((), dtype=torch.double))
