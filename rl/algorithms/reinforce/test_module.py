import torch

from rl.algorithms.reinforce.implementation import reinforce_update
from rl.policies.categorical.implementation import CategoricalPolicy


def test_rewarded_action_probability_increases() -> None:
    policy = CategoricalPolicy(2, 2)
    obs, actions = torch.ones(16, 2), torch.ones(16, dtype=torch.long)
    before = policy.distribution(obs).probs[:, 1].mean().detach()
    reinforce_update(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.1),
        obs,
        actions,
        torch.ones(16),
        normalize=False,
    )
    assert policy.distribution(obs).probs[:, 1].mean() > before
