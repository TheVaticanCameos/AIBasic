from copy import deepcopy

import torch
from torch import nn

from training.loop.implementation import clip_grad_norm, train_step


def test_unequal_microbatches_match_full_batch_update() -> None:
    full = nn.Linear(3, 1).double()
    micro = deepcopy(full)
    x, y = torch.randn(7, 3, dtype=torch.double), torch.randn(7, 1, dtype=torch.double)
    a = train_step(full, [(x, y)], torch.optim.SGD(full.parameters(), lr=0.1), nn.MSELoss())
    b = train_step(
        micro,
        [(x[:3], y[:3]), (x[3:6], y[3:6]), (x[6:], y[6:])],
        torch.optim.SGD(micro.parameters(), lr=0.1),
        nn.MSELoss(),
    )
    assert abs(a - b) < 1e-10
    for p, q in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(p, q)


def test_global_clipping_matches_torch() -> None:
    p, q = torch.ones(2, requires_grad=True), torch.ones(2, requires_grad=True)
    p.grad, q.grad = torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0])
    torch.testing.assert_close(clip_grad_norm([p], 2), torch.nn.utils.clip_grad_norm_([q], 2))
    torch.testing.assert_close(p.grad, q.grad)
