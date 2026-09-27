import torch

from rl.returns.discounted.implementation import discounted_returns
from rl.returns.n_step.implementation import n_step_returns
from rl.returns.td.implementation import td_target


def test_one_step_and_full_horizon_match_independent_algorithms() -> None:
    r, nv = torch.randn(6, 2), torch.randn(6, 2)
    term, trunc = torch.zeros(6, 2, dtype=torch.bool), torch.zeros(6, 2, dtype=torch.bool)
    term[2, 0], trunc[3, 1] = True, True
    torch.testing.assert_close(n_step_returns(r, nv, term, trunc, 1), td_target(r, nv, term))
    torch.testing.assert_close(
        n_step_returns(r, nv, term, trunc, 10), discounted_returns(r, term, trunc, nv)
    )


def test_two_step_hand_calculation() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    next_values = torch.tensor([[10.0], [20.0], [30.0]])
    terminated = torch.tensor([[False], [False], [True]])
    truncated = torch.zeros_like(terminated)
    actual = n_step_returns(rewards, next_values, terminated, truncated, n=2, gamma=0.5)
    # 第一个 target=1+0.5*2+0.25*20；第二个遇到 terminal，不 bootstrap。
    torch.testing.assert_close(actual, torch.tensor([[7.0], [3.5], [3.0]]))
