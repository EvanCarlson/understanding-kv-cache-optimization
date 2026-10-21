"""Correctness check every KV cache implementation must pass: decoding with
the cache must produce the exact same logits as recomputing the full prefix
from scratch at every step. If these diverge, the cache is reading or
writing the wrong slice somewhere.
"""

import pytest
import torch

from model import KVCacheFactory, ModelConfig, TinyTransformer


@pytest.mark.parametrize("cache_name", ["basic", "gqa", "paged", "quantized"])
def test_kv_cache_rejects_indivisible_head_dim(cache_name: str) -> None:
    config = ModelConfig(
        vocab_size=50,
        d_model=33,
        max_seq_len=16,
        n_layers=2,
        num_heads=4,
        num_kv_heads=2,
    )

    with pytest.raises(ValueError, match="divisible"):
        KVCacheFactory.create(
            cache_name, config=config, batch_size=1, device=torch.device("cpu")
        )


def test_basic_kv_cache_append_rejects_mismatched_shapes() -> None:
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        max_seq_len=16,
        n_layers=2,
        num_heads=4,
        num_kv_heads=4,
    )
    cache = KVCacheFactory.create(
        "basic", config=config, batch_size=1, device=torch.device("cpu")
    )

    keys = torch.zeros(1, 4, 2, 8)
    values = torch.zeros(1, 4, 3, 8)

    with pytest.raises(ValueError, match="identical shapes"):
        cache.append(layer_idx=0, keys=keys, values=values)


def test_basic_kv_cache_append_rejects_out_of_range_layer_idx() -> None:
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        max_seq_len=16,
        n_layers=2,
        num_heads=4,
        num_kv_heads=4,
    )
    cache = KVCacheFactory.create(
        "basic", config=config, batch_size=1, device=torch.device("cpu")
    )

    keys = torch.zeros(1, 4, 2, 8)
    values = torch.zeros(1, 4, 2, 8)

    with pytest.raises(IndexError, match="out of range"):
        cache.append(layer_idx=2, keys=keys, values=values)


@pytest.mark.parametrize("cache_name", ["basic", "gqa", "paged", "quantized"])
def test_cached_decode_matches_full_recompute(cache_name: str) -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        max_seq_len=16,
        n_layers=2,
        num_heads=4,
        num_kv_heads=4,
    )
    model = TinyTransformer(config=config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 4))
    max_new_tokens = 5

    cache = KVCacheFactory.create(
        cache_name, config=config, batch_size=1, device=torch.device("cpu")
    )

    # Decode with the cache, recording the logits used to pick each new
    # token along the way.
    generated = prompt
    cached_logits = []
    with torch.no_grad():
        step_input = prompt
        for _ in range(max_new_tokens):
            logits = model(step_input, cache=cache)
            next_logits = logits[:, -1, :]
            cached_logits.append(next_logits)

            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            step_input = next_token

    # For each of those same steps, recompute the full prefix from scratch
    # with no cache at all, and compare the last position's logits against
    # what the cache produced when it actually generated that step.
    prompt_len = prompt.shape[1]
    atol = 0.05 if cache_name == "quantized" else 1e-4
    with torch.no_grad():
        for step in range(max_new_tokens):
            prefix = generated[:, : prompt_len + step]
            full_logits = model(prefix, cache=None)
            recomputed_next_logits = full_logits[:, -1, :]

            assert torch.allclose(
                cached_logits[step], recomputed_next_logits, atol=atol
            )


@pytest.mark.parametrize("cache_name", ["basic", "gqa", "paged", "quantized"])
def test_chunked_prefill_matches_full_recompute_across_block_boundary(
    cache_name: str,
) -> None:
    """Split a prompt into uneven chunks straddling the paged cache's default
    128-token block size, to prove append()'s local causal mask (and the
    paged cache's block-splitting writes) are correct at those boundaries.
    """
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        max_seq_len=300,
        n_layers=2,
        num_heads=4,
        num_kv_heads=4,
    )
    model = TinyTransformer(config=config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 300))
    chunk_lengths = [127, 2, 129, 42]
    assert sum(chunk_lengths) == prompt.shape[1]

    cache = KVCacheFactory.create(
        cache_name, config=config, batch_size=1, device=torch.device("cpu")
    )

    chunk_logits = []
    with torch.no_grad():
        pos = 0
        for chunk_len in chunk_lengths:
            chunk = prompt[:, pos : pos + chunk_len]
            chunk_logits.append(model(chunk, cache=cache))
            pos += chunk_len

    cached_logits = torch.cat(chunk_logits, dim=1)

    with torch.no_grad():
        full_logits = model(prompt, cache=None)

    # Quantization noise compounds a little more over 4 differently-scaled
    # chunks and 300 tokens than in the short decode-equivalence test above.
    atol = 0.1 if cache_name == "quantized" else 1e-4
    assert torch.allclose(cached_logits, full_logits, atol=atol)


def test_advance_overflow_leaves_length_unchanged() -> None:
    config = ModelConfig(
        vocab_size=50,
        d_model=32,
        max_seq_len=16,
        n_layers=2,
        num_heads=4,
        num_kv_heads=4,
    )
    cache = KVCacheFactory.create(
        "basic", config=config, batch_size=1, device=torch.device("cpu")
    )
    cache.advance(num_new_tokens=10)

    with pytest.raises(RuntimeError, match="exceeded"):
        cache.advance(num_new_tokens=7)

    assert cache.length == 10
