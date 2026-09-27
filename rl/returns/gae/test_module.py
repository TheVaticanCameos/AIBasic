import torch

from rl.returns.gae.implementation import generalized_advantage_estimate


def test_hand_computed_gae_with_episode_boundary() -> None:
    r = torch.tensor([[1.0], [2.0], [100.0]])
    v = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    nv = torch.tensor([[20.0], [30.0], [40.0]], requires_grad=True)
    term, trunc = torch.tensor([[False], [False], [True]]), torch.tensor([[False], [True], [False]])
    advantage, returns = generalized_advantage_estimate(r, v, nv, term, trunc, 0.5, 1.0)
    torch.testing.assert_close(advantage, torch.tensor([[-0.5], [-3.0], [70.0]]))
    torch.testing.assert_close(returns, torch.tensor([[9.5], [17.0], [100.0]]))
    assert not advantage.requires_grad and not returns.requires_grad
    td_error, _ = generalized_advantage_estimate(r, v, nv, term, trunc, 0.5, 0.0)
    torch.testing.assert_close(td_error, torch.tensor([[1.0], [-3.0], [70.0]]))
