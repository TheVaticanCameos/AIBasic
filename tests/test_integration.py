import math

import pytest

from examples.supervised import train


@pytest.mark.parametrize("kind", ["mlp", "cnn", "transformer"])
def test_tiny_models_learn(kind: str) -> None:
    result = train(kind, steps=40)
    assert result["final_loss"] < result["initial_loss"] * 0.5
    assert result["training_accuracy"] > 0.9


@pytest.mark.parametrize("algorithm", ["dqn", "ppo", "reinforce", "actor_critic"])
def test_cartpole_collection_and_updates(algorithm: str) -> None:
    pytest.importorskip("gymnasium")
    from examples.cartpole import train as train_rl

    result = train_rl(algorithm, steps=160)
    assert result["updates"] > 0
    assert all(math.isfinite(v) for v in result.values())


@pytest.mark.parametrize("algorithm", ["ddpg", "td3", "sac"])
def test_pendulum_collection_scaling_and_updates(algorithm: str) -> None:
    pytest.importorskip("gymnasium")
    from examples.pendulum import train as train_rl

    result = train_rl(algorithm, steps=220)
    assert result["updates"] > 0 and result["completed_episodes"] == 1
    assert all(math.isfinite(v) for v in result.values())
