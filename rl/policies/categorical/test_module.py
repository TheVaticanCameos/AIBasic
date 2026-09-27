import torch

from rl.policies.categorical.implementation import CategoricalPolicy, ValueNetwork


def test_distribution_log_prob_entropy_and_singleton_batch() -> None:
    policy = CategoricalPolicy(3, 4)
    obs = torch.randn(1, 3)
    distribution = policy.distribution(obs)
    action = torch.tensor([2])
    torch.testing.assert_close(distribution.log_prob(action), policy(obs).log_softmax(-1)[:, 2])
    expected_entropy = -(distribution.probs * distribution.logits).sum(-1)
    torch.testing.assert_close(distribution.entropy(), expected_entropy)
    assert ValueNetwork(3)(obs).shape == (1,)
