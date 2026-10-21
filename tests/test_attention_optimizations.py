"""Check FP32 attention against the baseline math through prefill and cached decoding."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers.models.llama.modeling_llama import eager_attention_forward

from model import (
    BasicKVCache,
    GQAKVCache,
    ModelConfig,
    QuantizedKVCache,
    TinyMultiHeadAttention,
    TinyTransformer,
)


def make_config(num_kv_heads: int = 0) -> ModelConfig:
    return ModelConfig(41, 48, 16, 2, 12, num_kv_heads)


def make_cache(config: ModelConfig, batch_size: int):
    cache_type = GQAKVCache if config.num_kv_heads else BasicKVCache
    return cache_type(config=config, batch_size=batch_size, device=torch.device("cpu"))


def baseline_attend(queries, keys, values, *, cached: bool):
    query_len = queries.shape[2]
    group_idx = torch.arange(queries.shape[1]) // (queries.shape[1] // keys.shape[1])
    keys = keys[:, group_idx, :, :]
    values = values[:, group_idx, :, :]
    scores = torch.matmul(queries, keys.transpose(2, 3)) / queries.shape[-1] ** 0.5
    mask = torch.tril(torch.ones(query_len, query_len))
    if cached:
        if query_len > 1:
            scores[..., -query_len:] = scores[..., -query_len:].masked_fill(
                mask == 0, float("-inf")
            )
    else:
        scores = scores.masked_fill(mask == 0, -torch.inf)
    return torch.matmul(torch.softmax(scores, dim=-1), values)


@pytest.mark.parametrize("num_kv_heads", [0, 12, 3, 1])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("query_len,key_len", [(1, 1), (5, 5), (1, 8), (3, 8)])
@pytest.mark.parametrize("backend", ["expanded", "repeat_kv", "grouped"])
def test_attend_matches_published_math(
    num_kv_heads, batch_size, query_len, key_len, backend
):
    torch.manual_seed(13)
    config = replace(make_config(num_kv_heads), attention_backend=backend)
    attn = TinyMultiHeadAttention(config=config, layer_idx=0)
    queries = torch.randn(batch_size, query_len, 12, 4).transpose(1, 2)
    keys = torch.randn(batch_size, key_len, num_kv_heads or 12, 4).transpose(1, 2)
    values = torch.randn_like(keys)
    expected = baseline_attend(queries, keys, values, cached=key_len > query_len)
    actual = attn.attend(queries, keys, values)
    if backend == "expanded":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("num_kv_heads", [0, 12, 3, 1])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("backend", ["expanded", "repeat_kv", "grouped"])
@torch.no_grad()
def test_identical_parameters_prefill_decode_and_cached_chunks(
    num_kv_heads, batch_size, backend
):
    torch.manual_seed(21)
    config = make_config(num_kv_heads)
    baseline = TinyTransformer(config=config).eval()
    candidate_config = replace(config, attention_backend=backend)
    candidate = TinyTransformer(config=candidate_config).eval()
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    tokens = torch.randint(config.vocab_size, (batch_size, 10))
    expected_full = baseline(tokens)
    torch.testing.assert_close(candidate(tokens), expected_full, rtol=1e-5, atol=1e-5)
    baseline_cache = make_cache(config, batch_size)
    candidate_cache = make_cache(candidate_config, batch_size)
    outputs = []
    for start, end in [(0, 4), (4, 5), (5, 8), (8, 9), (9, 10)]:
        expected = baseline(tokens[:, start:end], cache=baseline_cache)
        actual = candidate(tokens[:, start:end], cache=candidate_cache)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        assert candidate_cache.length == baseline_cache.length == end
        outputs.append(actual)
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected_full, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("num_kv_heads", [0, 12, 3, 1])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("cached", [False, True])
@torch.no_grad()
def test_default_forward_matches_published_baseline_exactly(
    num_kv_heads, batch_size, cached
):
    torch.manual_seed(31)
    config = make_config(num_kv_heads)
    assert config.attention_backend == "expanded"
    attn = TinyMultiHeadAttention(config=config, layer_idx=0).eval()
    assert all(parameter.dtype == torch.float32 for parameter in attn.parameters())
    cache = make_cache(config, batch_size) if cached else None
    if cache is not None:
        assert cache.keys.dtype == cache.values.dtype == torch.float32
    previous_keys = []
    previous_values = []
    for seq_len in [4, 1, 3]:
        x = torch.randn(batch_size, seq_len, config.d_model)
        queries = attn.Q_proj(x).view(batch_size, seq_len, 12, 4).transpose(1, 2)
        keys = (
            attn.K_proj(x)
            .view(batch_size, seq_len, num_kv_heads or 12, 4)
            .transpose(1, 2)
        )
        values = (
            attn.V_proj(x)
            .view(batch_size, seq_len, num_kv_heads or 12, 4)
            .transpose(1, 2)
        )
        if cached:
            previous_keys.append(keys)
            previous_values.append(values)
            keys = torch.cat(previous_keys, dim=2)
            values = torch.cat(previous_values, dim=2)
        context = baseline_attend(queries, keys, values, cached=cached)
        expected = attn.out_proj(
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, config.d_model)
        )
        torch.testing.assert_close(attn(x, cache=cache), expected, rtol=0, atol=0)
        if cache is not None:
            cache.advance(num_new_tokens=seq_len)


def test_quantized_cache_storage_unchanged():
    config = make_config()
    cache = QuantizedKVCache(config=config, batch_size=1, device=torch.device("cpu"))
    assert cache.keys.dtype == cache.values.dtype == torch.uint8
    assert all(
        tensor.dtype == torch.float32
        for tensor in (cache.k_scale, cache.k_zero, cache.v_scale, cache.v_zero)
    )
    keys = torch.randn(1, 12, 3, 4)
    result = cache.append(layer_idx=0, keys=keys, values=keys)
    assert result.keys.dtype == result.values.dtype == torch.float32


@pytest.mark.parametrize("backend", ["", "invalid", "sdpa", "SDPA"])
def test_invalid_backend_rejected(backend):
    with pytest.raises(ValueError, match="Unknown attention backend"):
        replace(make_config(), attention_backend=backend)


@pytest.mark.parametrize("num_kv_heads", [0, 12, 3])
@torch.no_grad()
def test_attend_repeat_kv_copies_only_when_heads_are_shared(num_kv_heads):
    config = replace(make_config(num_kv_heads), attention_backend="repeat_kv")
    attn = TinyMultiHeadAttention(config=config, layer_idx=0)
    queries = torch.randn(1, 12, 1, 4)
    keys = torch.randn(1, num_kv_heads or 12, 5, 4)
    seen = []
    original = torch.matmul
    try:
        torch.matmul = lambda a, b: (seen.append(b), original(a, b))[1]
        attn.attend(queries, keys, keys)
    finally:
        torch.matmul = original
    shares_storage = seen[0].data_ptr() == keys.data_ptr()
    assert shares_storage == (num_kv_heads in (0, 12))


@pytest.mark.parametrize("num_kv_heads", [0, 12, 3, 1])
@pytest.mark.parametrize("query_len,key_len", [(1, 8), (5, 5), (3, 8)])
@torch.no_grad()
def test_repeat_kv_backend_matches_huggingface_eager_attention(
    num_kv_heads, query_len, key_len
):
    torch.manual_seed(17)
    kv_heads = num_kv_heads or 12
    config = replace(make_config(num_kv_heads), attention_backend="repeat_kv")
    attn = TinyMultiHeadAttention(config=config, layer_idx=0)
    queries = torch.randn(2, 12, query_len, 4)
    keys = torch.randn(2, kv_heads, key_len, 4)
    values = torch.randn_like(keys)
    mask = torch.zeros(query_len, key_len)
    for row in range(query_len):
        mask[row, key_len - query_len + row + 1 :] = float("-inf")
    module = SimpleNamespace(num_key_value_groups=12 // kv_heads, training=False)
    expected, _ = eager_attention_forward(
        module, queries, keys, values, mask[None, None], 4**-0.5
    )
    actual = attn.attend(queries, keys, values).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("keep", [1, 3, 10])
@torch.no_grad()
def test_forward_logits_to_keep_matches_tail_of_full_logits(keep):
    torch.manual_seed(5)
    model = TinyTransformer(config=make_config(12)).eval()
    tokens = torch.randint(model.input_embed.num_embeddings, (2, 10))
    full = model(tokens)
    kept = model(tokens, logits_to_keep=keep)
    assert kept.shape == (2, keep, full.shape[-1])
    torch.testing.assert_close(kept, full[:, -keep:, :], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("keep", [0, -1])
def test_forward_logits_to_keep_rejects_non_positive(keep):
    model = TinyTransformer(config=make_config(12)).eval()
    with pytest.raises(ValueError, match="logits_to_keep must be positive"):
        model(torch.zeros(1, 4, dtype=torch.long), logits_to_keep=keep)
