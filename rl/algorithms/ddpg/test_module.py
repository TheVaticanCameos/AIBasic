import torch

from rl.algorithms.ddpg.implementation import DDPG
from tests.rl_helpers import continuous_batch


def test_actor_updates_critic_frozen_during_actor_and_targets_detached() -> None:
    agent = DDPG(3, 2)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    target_before = [p.detach().clone() for p in agent.target_actor.parameters()]
    agent.update(continuous_batch())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert all(p.grad is None for p in agent.critic.parameters())
    assert all(p.grad is None for p in agent.target_critic.parameters())
    for old, online, target in zip(
        target_before, agent.actor.parameters(), agent.target_actor.parameters()
    ):
        torch.testing.assert_close(target, old * (1 - agent.tau) + online * agent.tau)
