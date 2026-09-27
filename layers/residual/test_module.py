import torch

from layers.residual.implementation import ResidualBlock


def test_identity_path_and_projection() -> None:
    block = ResidualBlock(3, 3).eval()
    with torch.no_grad():
        block.conv2.weight.zero_()
        block.conv2.bias.zero_()
    x = torch.randn(2, 3, 5, 5, requires_grad=True)
    torch.testing.assert_close(block(x), x.relu())
    projected = ResidualBlock(3, 8, stride=2)(x)
    assert projected.shape == (2, 8, 3, 3)
    projected.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
