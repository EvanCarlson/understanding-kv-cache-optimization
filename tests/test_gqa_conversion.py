"""Check mean-pooled GPT-2 head conversion, cached logits, and generation head options."""

from dataclasses import replace

import pytest
import torch

from generate import parse_args
from model import KVCacheFactory, ModelConfig, TinyTransformer, load_gpt2_weights


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=23,
        d_model=24,
        max_seq_len=10,
        n_layers=2,
        num_heads=12,
        num_kv_heads=12,
    )


@pytest.fixture
def checkpoint(config: ModelConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)

    def tensor(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.1

    d = config.d_model
    state = {
        "wte.weight": tensor(config.vocab_size, d),
        "wpe.weight": tensor(config.max_seq_len + 2, d),
        "ln_f.weight": tensor(d),
        "ln_f.bias": tensor(d),
    }
    for i in range(config.n_layers):
        prefix = f"h.{i}."
        for name, shape in {
            "ln_1.weight": (d,),
            "ln_1.bias": (d,),
            "ln_2.weight": (d,),
            "ln_2.bias": (d,),
            "attn.c_attn.weight": (d, 3 * d),
            "attn.c_attn.bias": (3 * d,),
            "attn.c_proj.weight": (d, d),
            "attn.c_proj.bias": (d,),
            "mlp.c_fc.weight": (d, 4 * d),
            "mlp.c_fc.bias": (4 * d,),
            "mlp.c_proj.weight": (4 * d, d),
            "mlp.c_proj.bias": (d,),
        }.items():
            state[prefix + name] = tensor(*shape)
    return state


@pytest.mark.parametrize("num_kv_heads", [0, 1, 3, 6, 12])
def test_checkpoint_conversion(
    config: ModelConfig,
    checkpoint: dict[str, torch.Tensor],
    num_kv_heads: int,
) -> None:
    model = TinyTransformer(config=replace(config, num_kv_heads=num_kv_heads))
    original = {name: value.clone() for name, value in checkpoint.items()}
    load_gpt2_weights(model, gpt2_state_dict=checkpoint)
    kv_heads = num_kv_heads or config.num_heads
    head_dim = config.d_model // config.num_heads
    group_size = config.num_heads // kv_heads

    expected = {
        "input_embed.weight": checkpoint["wte.weight"],
        "lm_head.weight": checkpoint["wte.weight"],
        "pos_embed.weight": checkpoint["wpe.weight"][: config.max_seq_len],
        "ln_f.weight": checkpoint["ln_f.weight"],
        "ln_f.bias": checkpoint["ln_f.bias"],
    }
    for i in range(config.n_layers):
        source = f"h.{i}."
        target = f"layers.{i}."
        for dest, src in {
            "ln_1": "ln_1",
            "ln_2": "ln_2",
            "attn.out_proj": "attn.c_proj",
            "ffn.fc1": "mlp.c_fc",
            "ffn.fc2": "mlp.c_proj",
        }.items():
            for suffix in ("weight", "bias"):
                value = checkpoint[source + src + "." + suffix]
                if value.ndim == 2:
                    value = value.T
                expected[target + dest + "." + suffix] = value

        for suffix in ("weight", "bias"):
            packed = checkpoint[source + "attn.c_attn." + suffix]
            if packed.ndim == 2:
                packed = packed.T
            for projection, value in zip("QKV", packed.chunk(3, dim=0)):
                if projection != "Q" and kv_heads != config.num_heads:
                    heads = value.split(head_dim, dim=0)
                    value = torch.cat(
                        [
                            torch.stack(heads[start : start + group_size]).mean(dim=0)
                            for start in range(0, config.num_heads, group_size)
                        ]
                    )
                expected[target + f"attn.{projection}_proj.{suffix}"] = value

    assert set(expected) == set(dict(model.named_parameters(remove_duplicate=False)))
    for name, parameter in model.named_parameters(remove_duplicate=False):
        pooled = kv_heads != config.num_heads and any(
            projection in name for projection in ("attn.K_proj.", "attn.V_proj.")
        )

        torch.testing.assert_close(
            parameter,
            expected[name],
            rtol=1e-6 if pooled else 0,
            atol=1e-8 if pooled else 0,
        )
    assert model.lm_head.weight is model.input_embed.weight
    for name, value in checkpoint.items():
        assert torch.equal(value, original[name])


@pytest.mark.parametrize("num_kv_heads", [1, 3, 12])
@pytest.mark.parametrize("backend", ["expanded", "grouped"])
def test_converted_cached_logits_match_uncached(
    config: ModelConfig,
    checkpoint: dict[str, torch.Tensor],
    num_kv_heads: int,
    backend: str,
) -> None:
    config = replace(config, num_kv_heads=num_kv_heads, attention_backend=backend)
    model = TinyTransformer(config=config).eval()
    load_gpt2_weights(model, gpt2_state_dict=checkpoint)
    cache = KVCacheFactory.create(
        "gqa", config=config, batch_size=2, device=torch.device("cpu")
    )
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    with torch.no_grad():
        full = model(tokens)

        cached = torch.cat(
            [
                model(tokens[:, :3], cache=cache),
                model(tokens[:, 3:5], cache=cache),
                model(tokens[:, 5:], cache=cache),
            ],
            dim=1,
        )
    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)
    assert cache.length == tokens.shape[1]


@pytest.mark.parametrize("cache_type", ["basic", "gqa", "paged", "quantized"])
def test_generation_default_heads(cache_type: str) -> None:
    args = parse_args(["--cache-type", cache_type])
    assert args.num_kv_heads == (3 if cache_type == "gqa" else 12)


@pytest.mark.parametrize("heads", [1, 2, 3, 4, 6, 12])
def test_generation_explicit_gqa_heads(heads: int) -> None:
    args = parse_args(["--cache-type", "gqa", "--num-kv-heads", str(heads)])
    assert args.num_kv_heads == heads


@pytest.mark.parametrize("heads", [0, -1, 5, 13, 24])
def test_generation_rejects_invalid_heads(heads: int, capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        parse_args(["--cache-type", "gqa", "--num-kv-heads", str(heads)])
    assert "positive divisor of 12" in capsys.readouterr().err


@pytest.mark.parametrize("cache_type", ["basic", "paged", "quantized"])
@pytest.mark.parametrize("heads", [1, 3, 12])
def test_generation_non_gqa_requires_mha(cache_type: str, heads: int, capsys) -> None:
    argv = ["--cache-type", cache_type, "--num-kv-heads", str(heads)]
    if heads == 12:
        assert parse_args(argv).num_kv_heads == 12
    else:
        with pytest.raises(SystemExit, match="2"):
            parse_args(argv)
        assert "requires --cache-type gqa" in capsys.readouterr().err
