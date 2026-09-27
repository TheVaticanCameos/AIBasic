import torch

from attention.scaled_dot_product.implementation import causal_mask
from attention.transformer.implementation import TransformerBlock


def test_residual_identity_and_causality() -> None:
    model = TransformerBlock(8, 2, 16).eval()
    x = torch.randn(2, 4, 8)
    changed = x.clone()
    changed[:, 2:] += 5
    torch.testing.assert_close(
        model(x, causal_mask(4, 4))[:, :2], model(changed, causal_mask(4, 4))[:, :2]
    )
    with torch.no_grad():
        model.attention.out_proj.weight.zero_()
        model.attention.out_proj.bias.zero_()
        model.ffn.down.weight.zero_()
        model.ffn.down.bias.zero_()
    torch.testing.assert_close(model(x), x)
