"""REINFORCE 的一次更新；采样与 return 计算由外层负责。"""

import torch
from torch import Tensor

from rl.objectives.policy_gradient.implementation import policy_gradient_loss
from rl.policies.categorical.implementation import CategoricalPolicy
from rl.returns.td.implementation import normalize_advantages


def reinforce_update(
    policy: CategoricalPolicy,
    optimizer: torch.optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    returns: Tensor,
    normalize: bool = True,
) -> float:
    """obs [N,O], actions int64 [N], returns [N] -> loss；N 为轨迹总步数。

    标准化是可选 variance-reduction 启发式；小 batch 下可能抹去学习信号。
    """
    weights = normalize_advantages(returns) if normalize else returns
    distribution = policy.distribution(observations)
    loss = policy_gradient_loss(distribution.log_prob(actions), weights)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


if __name__ == "__main__":
    policy = CategoricalPolicy(4, 2)
    optimizer = torch.optim.Adam(policy.parameters())
    print(
        reinforce_update(
            policy, optimizer, torch.randn(4, 4), torch.tensor([0, 1, 0, 1]), torch.arange(4.0)
        )
    )
