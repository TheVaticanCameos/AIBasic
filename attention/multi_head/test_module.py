import torch
from torch import nn

from attention.multi_head.implementation import MultiHeadAttention
from attention.scaled_dot_product.implementation import attention
from fundamentals.tensor_ops.implementation import merge_heads, split_heads


def test_mha_matches_torch_cross_attention() -> None:
    own, ref = (
        MultiHeadAttention(8, 2).double(),
        nn.MultiheadAttention(8, 2, batch_first=True).double(),
    )
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([own.q_proj.weight, own.k_proj.weight, own.v_proj.weight])
        )
        ref.in_proj_bias.copy_(torch.cat([own.q_proj.bias, own.k_proj.bias, own.v_proj.bias]))
        ref.out_proj.load_state_dict(own.out_proj.state_dict())
    x, context = torch.randn(2, 3, 8, dtype=torch.double), torch.randn(2, 5, 8, dtype=torch.double)
    valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    expected, _ = ref(x, context, context, key_padding_mask=~valid)
    torch.testing.assert_close(own(x, context, valid[:, None, None, :]), expected)


def test_gqa_matches_explicit_per_head_selection() -> None:
    for kv_heads in (1, 2):
        model = MultiHeadAttention(16, 4, kv_heads)
        x = torch.randn(2, 3, 16)
        q = split_heads(model.q_proj(x), 4)
        k, v = split_heads(model.k_proj(x), kv_heads), split_heads(model.v_proj(x), kv_heads)
        outputs = [
            attention(q[:, i], k[:, i // (4 // kv_heads)], v[:, i // (4 // kv_heads)])[0]
            for i in range(4)
        ]
        expected = model.out_proj(merge_heads(torch.stack(outputs, dim=1)))
        torch.testing.assert_close(model(x), expected)
