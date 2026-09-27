import math

import torch

from rl.algorithms.sac.implementation import SAC
from tests.rl_helpers import continuous_batch


def test_sac_updates_actor_and_temperature_without_target_gradients() -> None:
    agent = SAC(3, 2)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    alpha = agent.alpha.clone()
    stats = agent.update(continuous_batch())
    assert all(math.isfinite(v) for v in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert not torch.equal(alpha, agent.alpha)
    assert agent.alpha > 0
    assert all(p.grad is None for p in agent.target_q1.parameters())
    assert all(p.grad is None for p in agent.q1.parameters())


def test_fixed_temperature_and_terminal_targets() -> None:
    agent = SAC(3, 2, automatic_temperature=False)
    batch = continuous_batch()
    targets = agent.targets(batch)
    torch.testing.assert_close(targets[batch.terminated], batch.rewards[batch.terminated])
    alpha = agent.alpha.clone()
    agent.update(batch)
    torch.testing.assert_close(agent.alpha, alpha)


def test_soft_target_uses_minimum_q_and_subtracts_log_prob() -> None:
    agent = SAC(3, 2, gamma=0.5, alpha=0.2)
    batch = continuous_batch()
    with torch.no_grad():
        for network, bias in ((agent.target_q1, 3.0), (agent.target_q2, 9.0)):
            for parameter in network.parameters():
                parameter.zero_()
            network.network[-1].bias.fill_(bias)
        torch.manual_seed(123)
        _, log_prob = agent.actor.sample(batch.next_observations)
    expected = batch.rewards + 0.5 * (~batch.terminated) * (3.0 - 0.2 * log_prob)
    torch.manual_seed(123)
    actual = agent.targets(batch)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad
