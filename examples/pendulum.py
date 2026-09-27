"""Pendulum 的 DDPG/TD3/SAC 示例；环境缩放只发生在采样边界。"""

import argparse
import json

import gymnasium as gym
import numpy as np
import torch

from rl.algorithms.ddpg.implementation import DDPG
from rl.algorithms.sac.implementation import SAC
from rl.algorithms.td3.implementation import TD3
from rl.buffers.replay.implementation import ReplayBuffer


def train(algorithm: str = "sac", steps: int = 3000, seed: int = 0) -> dict[str, float]:
    """buffer 始终保存 [-1,1] 动作，交互时映射到 Pendulum 的 [-2,2]。"""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    factories = {"ddpg": DDPG, "td3": TD3, "sac": SAC}
    if algorithm not in factories or steps < 1:
        raise ValueError("invalid algorithm or steps")
    agent = factories[algorithm](3, 1)
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    assert isinstance(env.action_space, gym.spaces.Box)
    low, high = env.action_space.low, env.action_space.high
    buffer = ReplayBuffer(10000, 3, 1)
    returns: list[float] = []
    score, updates = 0.0, 0
    try:
        for step in range(steps):
            observation = torch.as_tensor(obs)  # [3]
            if step < 128:
                normalized = torch.as_tensor(
                    2 * (env.action_space.sample() - low) / (high - low) - 1
                )
            elif isinstance(agent, SAC):
                normalized = agent.act(observation[None])[0]  # [1]
            else:
                normalized = agent.act(observation[None], noise_std=0.1)[0]
            action = low + (normalized.numpy() + 1) * 0.5 * (high - low)  # [1]，环境动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(
                observation,
                normalized,
                float(reward),
                torch.as_tensor(next_obs),
                terminated,
                truncated,
            )
            if len(buffer) >= 128:
                agent.update(buffer.sample(64))
                updates += 1
            score += float(reward)
            if terminated or truncated:
                returns.append(score)
                score = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
    finally:
        env.close()
    return {
        "steps": float(steps),
        "updates": float(updates),
        "completed_episodes": float(len(returns)),
        "recent_train_return": float(np.mean(returns[-10:])) if returns else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["ddpg", "td3", "sac"], default="sac")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args.algorithm, args.steps, args.seed), indent=2))
