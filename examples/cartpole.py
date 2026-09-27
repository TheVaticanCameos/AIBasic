"""Gymnasium CartPole 示例：DQN/PPO/REINFORCE/一步 Actor-Critic。"""

import argparse
import json
from collections.abc import Callable

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from rl.algorithms.actor_critic.implementation import actor_critic_update
from rl.algorithms.dqn.implementation import DQN
from rl.algorithms.ppo.implementation import PPO
from rl.algorithms.reinforce.implementation import reinforce_update
from rl.buffers.replay.implementation import ReplayBuffer, TransitionBatch
from rl.buffers.rollout.implementation import RolloutBatch, RolloutBuffer
from rl.policies.categorical.implementation import CategoricalPolicy, ValueNetwork
from rl.returns.discounted.implementation import discounted_returns


def evaluate(action_fn: Callable[[Tensor], Tensor], seed: int, episodes: int = 3) -> float:
    """action_fn: obs [1,4] -> action [1]；独立环境上的确定性评估。"""
    env = gym.make("CartPole-v1")
    returns: list[float] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            score = 0.0
            while True:
                with torch.no_grad():
                    action = int(action_fn(torch.as_tensor(obs).unsqueeze(0)).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                if terminated or truncated:
                    break
            returns.append(score)
    finally:
        env.close()
    return float(np.mean(returns))


def train(algorithm: str = "dqn", steps: int = 5000, seed: int = 0) -> dict[str, float]:
    """固定交互步数；报告回报仅作运行观察，短训练不保证收敛。"""
    if algorithm not in {"dqn", "ppo", "reinforce", "actor_critic"} or steps < 1:
        raise ValueError("invalid algorithm or steps")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    dqn = DQN(4, 2) if algorithm == "dqn" else None
    ppo = PPO(4, 2) if algorithm == "ppo" else None
    policy, value = CategoricalPolicy(4, 2), ValueNetwork(4)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    replay, rollout = ReplayBuffer(10000, 4), RolloutBuffer()
    trajectory_obs: list[Tensor] = []
    trajectory_actions: list[Tensor] = []
    trajectory_rewards: list[float] = []
    episode_returns: list[float] = []
    score, updates = 0.0, 0
    try:
        for step in range(steps):
            observation = torch.as_tensor(obs).unsqueeze(0)  # [1,4]
            with torch.no_grad():
                if dqn is not None:
                    action = dqn.act(observation, epsilon=max(0.05, 1 - step / max(steps * 0.6, 1)))
                elif ppo is not None:
                    action, old_log_prob, old_value = ppo.act(observation)
                else:
                    action = policy.distribution(observation).sample()  # [1]
            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            next_observation = torch.as_tensor(next_obs).unsqueeze(0)  # reset 前 [1,4]
            r = float(reward)
            term, trunc = torch.tensor([terminated]), torch.tensor([truncated])
            if dqn is not None:
                replay.add(observation[0], action[0], r, next_observation[0], terminated, truncated)
                if len(replay) >= 64:
                    dqn.update(replay.sample(64))
                    updates += 1
            elif ppo is not None:
                with torch.no_grad():
                    next_value = ppo.value(next_observation)  # [1]
                rollout.add(
                    RolloutBatch(
                        observation,
                        action,
                        torch.tensor([r]),
                        old_value,
                        next_value,
                        old_log_prob,
                        term,
                        trunc,
                    )
                )
                if (step + 1) % 128 == 0 or step == steps - 1:
                    ppo.update(rollout.finish())
                    updates += 1
            elif algorithm == "actor_critic":
                batch = TransitionBatch(
                    observation, action, torch.tensor([r]), next_observation, term, trunc
                )
                actor_critic_update(policy, value, policy_optimizer, value_optimizer, batch)
                updates += 1
            else:
                trajectory_obs.append(observation[0])
                trajectory_actions.append(action[0])
                trajectory_rewards.append(r)
                if terminated or truncated:
                    rewards = torch.tensor(trajectory_rewards)[:, None]  # [T,1]
                    flags = torch.zeros_like(rewards, dtype=torch.bool)
                    # 此 REINFORCE 示例优化有限 episode 的 MC 回报，无 learned bootstrap。
                    returns = discounted_returns(rewards, flags, flags, torch.zeros_like(rewards))
                    reinforce_update(
                        policy,
                        policy_optimizer,
                        torch.stack(trajectory_obs),
                        torch.stack(trajectory_actions),
                        returns[:, 0],
                    )
                    trajectory_obs.clear()
                    trajectory_actions.clear()
                    trajectory_rewards.clear()
                    updates += 1
            score += r
            if terminated or truncated:
                episode_returns.append(score)
                score = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
    finally:
        env.close()

    def greedy_action(observations: Tensor) -> Tensor:
        if dqn is not None:
            return dqn.act(observations)
        if ppo is not None:
            return ppo.policy(observations).argmax(-1)
        return policy(observations).argmax(-1)

    return {
        "steps": float(steps),
        "updates": float(updates),
        "completed_episodes": float(len(episode_returns)),
        "recent_train_return": float(np.mean(episode_returns[-10:])) if episode_returns else 0.0,
        "eval_return": evaluate(greedy_action, seed + 1000),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=["dqn", "ppo", "reinforce", "actor_critic"], default="dqn"
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(train(args.algorithm, args.steps, args.seed), indent=2))
