import pytest
import torch

from attention.kv_cache.implementation import KVCache, cached_attention
from attention.multi_head.implementation import MultiHeadAttention
from attention.scaled_dot_product.implementation import causal_mask


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_chunked_decoding_matches_full_sequence(kv_heads: int) -> None:
    model = MultiHeadAttention(16, 4, kv_heads).eval()
    x, cache = torch.randn(2, 7, 16), KVCache()
    expected = model(x, mask=causal_mask(7, 7))
    actual = torch.cat(
        [cached_attention(model, chunk, cache) for chunk in x.split([3, 1, 3], dim=1)], dim=1
    )
    torch.testing.assert_close(actual, expected)
    assert cache.keys is not None and not cache.keys.requires_grad
    cache.clear()
    assert cache.keys is cache.values is None
    with pytest.raises(ValueError):
        cached_attention(model.train(), x, cache)
