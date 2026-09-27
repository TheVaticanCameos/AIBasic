import torch
from torch.distributions import Independent, Normal, TanhTransform, TransformedDistribution

from rl.policies.gaussian.implementation import GaussianPolicy


def test_tanh_density_matches_transformed_distribution_and_rsample_gradient() -> None:
    policy = GaussianPolicy(3, 2).double()
    obs = torch.randn(4, 3, dtype=torch.double)
    actions, log_probs = policy.sample(obs)
    mean, log_std = policy(obs)
    expected_distribution = TransformedDistribution(
        Independent(Normal(mean, log_std.exp()), 1), [TanhTransform(cache_size=1)]
    )
    torch.testing.assert_close(log_probs, expected_distribution.log_prob(actions))
    actions.sum().backward()
    assert policy.mean.weight.grad is not None and policy.mean.weight.grad.abs().sum() > 0


def test_extreme_mean_has_finite_corrected_log_prob() -> None:
    policy = GaussianPolicy(3, 2)
    with torch.no_grad():
        policy.mean.weight.zero_()
        policy.mean.bias.fill_(100)
    actions, log_probs = policy.sample(torch.zeros(2, 3), deterministic=True)
    assert torch.isfinite(log_probs).all() and (actions.abs() <= 1).all()
