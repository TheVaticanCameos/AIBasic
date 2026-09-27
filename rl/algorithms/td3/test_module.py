import torch

from rl.algorithms.td3.implementation import TD3
from tests.rl_helpers import continuous_batch


def test_delayed_actor_and_target_updates() -> None:
    agent = TD3(3, 2, policy_delay=2)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    targets = [p.detach().clone() for p in agent.target_q1.parameters()]
    batch = continuous_batch()
    assert agent.update(batch)["actor_loss"] is None
    assert all(torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(targets, agent.target_q1.parameters()))
    assert agent.update(batch)["actor_loss"] is not None
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert all(p.grad is None for p in agent.target_q1.parameters())


def test_clipped_double_q_target_and_terminal_mask() -> None:
    agent = TD3(3, 2, gamma=0.5, target_noise=0.0)
    with torch.no_grad():
        for network, value in ((agent.target_q1, 3.0), (agent.target_q2, 9.0)):
            for p in network.parameters():
                p.zero_()
            network.network[-1].bias.fill_(value)
    batch = continuous_batch()
    expected = batch.rewards + (~batch.terminated) * 1.5
    torch.testing.assert_close(agent.targets(batch), expected)
