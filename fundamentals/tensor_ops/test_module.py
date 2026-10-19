import pytest
import torch

from fundamentals.tensor_ops.implementation import (
    image_to_tokens,
    merge_heads,
    pairwise_dot,
    select_action_values,
    split_heads,
)


def test_heads_roundtrip_noncontiguous_and_gather() -> None:
    x = torch.randn(2, 8, 3).transpose(1, 2)
    torch.testing.assert_close(merge_heads(split_heads(x, 4)), x)
    q = torch.tensor([[1.0, 5.0], [9.0, 2.0]], requires_grad=True)
    selected = select_action_values(q, torch.tensor([1, 0]))
    selected.sum().backward()
    torch.testing.assert_close(q.grad, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    y = torch.randn(2, 5, 8)
    torch.testing.assert_close(pairwise_dot(x, y), x @ y.transpose(-2, -1))
    with pytest.raises(ValueError):
        split_heads(x, 3)


def test_permute_preserves_spatial_order_and_channels() -> None:
    image = torch.arange(12.0).reshape(1, 2, 2, 3)
    tokens = image_to_tokens(image)
    expected = torch.tensor(
        [[[0.0, 6.0], [1.0, 7.0], [2.0, 8.0], [3.0, 9.0], [4.0, 10.0], [5.0, 11.0]]]
    )
    torch.testing.assert_close(tokens, expected)


@pytest.mark.parametrize("num_heads", [0, -1, 3])
def test_split_heads_rejects_nonpositive_or_nondivisible_heads(num_heads: int) -> None:
    with pytest.raises(ValueError):
        split_heads(torch.randn(2, 3, 8), num_heads)
