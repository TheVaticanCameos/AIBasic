import math

import torch

from rl.algorithms.ppo.implementation import PPO, PPOConfig
from rl.buffers.rollout.implementation import RolloutBatch


def test_multiepoch_update_preserves_old_data_and_updates_actor_value() -> None:
    agent = PPO(3, 2, PPOConfig(epochs=2, minibatch_size=3))
    obs = torch.randn(5, 2, 3)
    action, log_prob, value = agent.act(obs.flatten(0, 1))
    old = log_prob.detach().clone()
    rollout = RolloutBatch(
        obs,
        action.reshape(5, 2),
        torch.randn(5, 2),
        value.reshape(5, 2),
        torch.zeros(5, 2),
        log_prob.reshape(5, 2),
        torch.zeros(5, 2, dtype=torch.bool),
        torch.zeros(5, 2, dtype=torch.bool),
    )
    actor_before = [p.detach().clone() for p in agent.policy.parameters()]
    value_before = [p.detach().clone() for p in agent.value.parameters()]
    stats = agent.update(rollout)
    assert all(math.isfinite(v) for v in stats.values())
    torch.testing.assert_close(rollout.log_probs.flatten(), old)
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.policy.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(value_before, agent.value.parameters()))
    assert not rollout.values.requires_grad and not rollout.log_probs.requires_grad
