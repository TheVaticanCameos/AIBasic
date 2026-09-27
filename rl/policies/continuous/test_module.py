import torch

from rl.policies.continuous.implementation import ContinuousQ, DeterministicPolicy


def test_actor_bounds_and_action_to_value_gradient() -> None:
    actor, critic = DeterministicPolicy(3, 2), ContinuousQ(3, 2)
    obs = torch.randn(4, 3)
    actions = actor(obs)
    assert (actions.abs() <= 1).all()
    gradient = torch.autograd.grad(critic(obs, actions).sum(), actions)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
