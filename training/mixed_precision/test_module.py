import math

import torch
from torch import nn

from training.mixed_precision.implementation import amp_train_step


def test_cpu_amp_updates_parameters() -> None:
    model = nn.Linear(3, 1)
    before = model.weight.detach().clone()
    loss = amp_train_step(
        model,
        torch.randn(5, 3),
        torch.randn(5, 1),
        torch.optim.SGD(model.parameters(), lr=0.1),
        nn.MSELoss(),
        torch.amp.GradScaler("cuda", enabled=False),
        max_grad_norm=1.0,
    )
    assert math.isfinite(loss)
    assert not torch.equal(model.weight, before)
