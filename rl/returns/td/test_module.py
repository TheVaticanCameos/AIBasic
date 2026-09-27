import torch

from rl.returns.td.implementation import normalize_advantages, td_target


def test_bootstrap_and_gradient_boundary() -> None:
    result = td_target(
        torch.tensor([1.0, 1.0]),
        torch.tensor([10.0, 10.0], requires_grad=True),
        torch.tensor([True, False]),
        gamma=0.9,
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 10.0]))
    assert not result.requires_grad
    torch.testing.assert_close(normalize_advantages(torch.ones(1)), torch.zeros(1))
    normalized = normalize_advantages(torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(normalized.std(correction=0), torch.tensor(1.0))
