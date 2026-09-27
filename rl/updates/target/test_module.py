import pytest
import torch
from torch import nn

from rl.updates.target.implementation import frozen_parameters, hard_update, soft_update


def test_polyak_and_hard_copy() -> None:
    source, target = nn.Linear(2, 1), nn.Linear(2, 1)
    previous = target.weight.detach().clone()
    soft_update(target, source, 0.25)
    torch.testing.assert_close(target.weight, 0.75 * previous + 0.25 * source.weight)
    hard_update(target, source)
    torch.testing.assert_close(target.weight, source.weight)


def test_freezing_preserves_input_gradient_and_restores_after_exception() -> None:
    model = nn.Linear(2, 1)
    x = torch.ones(1, 2, requires_grad=True)
    with frozen_parameters(model):
        model(x).sum().backward()
    assert x.grad is not None and model.weight.grad is None
    with pytest.raises(RuntimeError), frozen_parameters(model):
        raise RuntimeError("test restoration")
    assert all(p.requires_grad for p in model.parameters())
