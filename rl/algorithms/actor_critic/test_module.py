import torch

from rl.algorithms.actor_critic.implementation import actor_critic_update
from rl.buffers.replay.implementation import TransitionBatch
from rl.policies.categorical.implementation import CategoricalPolicy, ValueNetwork


def test_terminal_reward_learning_and_no_next_obs_gradient() -> None:
    policy, value = CategoricalPolicy(2, 2), ValueNetwork(2)
    obs = torch.ones(8, 2)
    next_obs = torch.randn(8, 2, requires_grad=True)
    batch = TransitionBatch(
        obs,
        torch.zeros(8, dtype=torch.long),
        torch.ones(8),
        next_obs,
        torch.ones(8, dtype=torch.bool),
        torch.zeros(8, dtype=torch.bool),
    )
    before = (value(obs) - 1).square().mean().detach()
    actor_critic_update(
        policy,
        value,
        torch.optim.SGD(policy.parameters(), lr=0.01),
        torch.optim.SGD(value.parameters(), lr=0.01),
        batch,
    )
    assert (value(obs) - 1).square().mean() < before
    assert next_obs.grad is None
