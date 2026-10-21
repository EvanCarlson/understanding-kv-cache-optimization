"""Check FP32 attention options and benchmark metadata without remote downloads."""

from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

import benchmark
import generate
from model import ModelConfig


@pytest.fixture
def offline_config(*, monkeypatch: pytest.MonkeyPatch) -> ModelConfig:
    config = ModelConfig(
        vocab_size=23,
        d_model=24,
        max_seq_len=8,
        n_layers=1,
        num_heads=12,
        num_kv_heads=3,
    )
    d = config.d_model
    shapes = {
        "wte.weight": (config.vocab_size, d),
        "wpe.weight": (config.max_seq_len, d),
        "ln_f.weight": (d,),
        "ln_f.bias": (d,),
        "h.0.ln_1.weight": (d,),
        "h.0.ln_1.bias": (d,),
        "h.0.ln_2.weight": (d,),
        "h.0.ln_2.bias": (d,),
        "h.0.attn.c_attn.weight": (d, 3 * d),
        "h.0.attn.c_attn.bias": (3 * d,),
        "h.0.attn.c_proj.weight": (d, d),
        "h.0.attn.c_proj.bias": (d,),
        "h.0.mlp.c_fc.weight": (d, 4 * d),
        "h.0.mlp.c_fc.bias": (4 * d,),
        "h.0.mlp.c_proj.weight": (4 * d, d),
        "h.0.mlp.c_proj.bias": (d,),
    }
    weights = {name: torch.full(shape, 0.125) for name, shape in shapes.items()}

    def download(*, repo_id: str, filename: str) -> str:
        assert repo_id == "offline"
        assert filename == "model.safetensors"
        return "offline.safetensors"

    def read_weights(path: str, *, device: str) -> dict[str, torch.Tensor]:
        assert path == "offline.safetensors"
        assert device == "cpu"
        return weights

    monkeypatch.setattr(generate, "hf_hub_download", download)
    monkeypatch.setattr(generate, "load_file", read_weights)
    monkeypatch.setattr(benchmark, "GPT2_SMALL_D_MODEL", config.d_model)
    monkeypatch.setattr(benchmark, "GPT2_SMALL_N_LAYERS", config.n_layers)
    monkeypatch.setattr(
        benchmark.tiktoken,
        "get_encoding",
        lambda _: SimpleNamespace(n_vocab=config.vocab_size, eot_token=-1),
    )
    return config


@pytest.mark.parametrize("module", [generate, benchmark])
@pytest.mark.parametrize("cache_type", ["basic", "gqa", "paged", "quantized"])
def test_parse_args_baseline_defaults(*, module: ModuleType, cache_type: str) -> None:
    args = module.parse_args(["--cache-type", cache_type])
    assert args.attention_backend == "expanded"
    assert not hasattr(args, "dtype")
    assert args.num_kv_heads == (3 if cache_type == "gqa" else 12)
    assert args.seed == benchmark.DEFAULT_SEED


@pytest.mark.parametrize(
    "module,cache_type",
    [(generate, "basic"), (generate, "gqa")]
    + [(benchmark, cache) for cache in ("basic", "gqa", "none")],
)
@pytest.mark.parametrize("backend", ["expanded", "grouped"])
def test_parse_args_attention_options(
    *, module: ModuleType, cache_type: str, backend: str
) -> None:
    args = module.parse_args(
        ["--cache-type", cache_type, "--attention-backend", backend, "--seed", "7"]
    )
    assert args.attention_backend == backend
    assert args.seed == 7


@pytest.mark.parametrize("module", [generate, benchmark])
@pytest.mark.parametrize("cache_type", ["paged", "quantized"])
def test_parse_args_unsupported_cache_options(
    *, module: ModuleType, cache_type: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        module.parse_args(
            ["--cache-type", cache_type, "--attention-backend", "grouped"]
        )
    assert "require basic, gqa, or none" in capsys.readouterr().err


@pytest.mark.parametrize("module", [generate, benchmark])
@pytest.mark.parametrize(
    "options",
    [
        ["--attention-backend", "sdpa"],
        ["--attention-backend", "flash"],
        ["--dtype", "float32"],
        ["--dtype", "bfloat16"],
        ["--dtype", "float16"],
        ["--compile"],
    ],
)
def test_parse_args_removed_or_invalid_options(
    *, module: ModuleType, options: list[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        module.parse_args(options)


def test_generate_exports_only_retained_backends() -> None:
    assert generate.ATTENTION_BACKEND_CHOICES == ["expanded", "grouped"]
    assert not hasattr(generate, "DTYPES")


@pytest.mark.parametrize("heads", [1, 2, 3, 4, 6, 12])
def test_benchmark_parse_args_explicit_gqa_heads(*, heads: int) -> None:
    args = benchmark.parse_args(["--cache-type", "gqa", "--num-kv-heads", str(heads)])
    assert args.num_kv_heads == heads


@pytest.mark.parametrize("heads", [0, -1, 5, 13, 24])
def test_benchmark_parse_args_invalid_heads(*, heads: int) -> None:
    with pytest.raises(SystemExit, match="2"):
        benchmark.parse_args(["--cache-type", "gqa", "--num-kv-heads", str(heads)])


@pytest.mark.parametrize("cache_type", ["basic", "paged", "quantized", "none"])
def test_benchmark_parse_args_reduced_heads_require_gqa(*, cache_type: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        benchmark.parse_args(["--cache-type", cache_type, "--num-kv-heads", "3"])


@pytest.mark.parametrize("backend", ["expanded", "grouped"])
def test_load_pretrained_model_returns_fp32(
    *, offline_config: ModelConfig, backend: str
) -> None:
    config = replace(offline_config, attention_backend=backend)
    assert not hasattr(config, "dtype")
    model = generate.load_pretrained_model(
        config=config, model_repo="offline", device=torch.device("cpu")
    )
    assert not model.training
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        assert torch.all(parameter == 0.125)


@pytest.mark.parametrize(
    "cache_type,backend",
    [
        (cache, backend)
        for cache in ("basic", "gqa", "none")
        for backend in ("expanded", "grouped")
    ]
    + [(cache, "expanded") for cache in ("paged", "quantized")],
)
def test_run_benchmark_config_and_metadata(
    *, offline_config: ModelConfig, cache_type: str, backend: str
) -> None:
    args = benchmark.parse_args(
        ["--cache-type", cache_type, "--attention-backend", backend]
    )
    result, trace = benchmark.run_benchmark(
        cache_type=cache_type,
        context_lengths=[2],
        batch_sizes=[1],
        num_trials=1,
        max_new_tokens=1,
        model_repo="offline",
        gpu_type="cpu",
        device=torch.device("cpu"),
        capture_trace=False,
        seed=7,
        num_kv_heads=args.num_kv_heads,
        attention_backend=backend,
    )
    assert result.attention_backend == backend
    assert result.dtype == "float32"
    assert result.num_kv_heads == args.num_kv_heads
    assert result.seed == 7
    assert result.model_repo == "offline"
    assert result.max_seq_len == 4
    assert result.max_new_tokens == 1
    assert result.torch_version == torch.__version__
    assert result.points[0].timed_steps == 1
    assert not result.points[0].oom
    assert trace is None
    if cache_type in ("basic", "gqa"):
        head_dim = offline_config.d_model // offline_config.num_heads
        assert (
            result.points[0].cache_bytes_used
            == 2 * args.num_kv_heads * head_dim * 2 * 4
        )
        assert (
            result.points[0].cache_bytes_allocated
            == 2 * result.points[0].cache_bytes_used
        )
    elif cache_type == "none":
        assert result.points[0].cache_bytes_used == 0
        assert result.points[0].cache_bytes_allocated == 0


@pytest.mark.parametrize(
    "cache_type,backend,message",
    [
        ("paged", "grouped", "require basic, gqa, or none"),
        ("quantized", "grouped", "require basic, gqa, or none"),
        ("basic", "sdpa", "Unknown attention backend"),
    ],
)
def test_run_benchmark_rejects_unsupported_options(
    *, cache_type: str, backend: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.run_benchmark(
            cache_type=cache_type,
            context_lengths=[2],
            batch_sizes=[1],
            num_trials=1,
            max_new_tokens=1,
            model_repo="offline",
            gpu_type="cpu",
            device=torch.device("cpu"),
            capture_trace=False,
            seed=42,
            attention_backend=backend,
        )


@pytest.mark.parametrize(
    "cache_type,heads,key",
    [
        ("basic", 12, "basic"),
        ("gqa", 3, "gqa_h3"),
        ("gqa", 1, "gqa_h1"),
        ("none", 12, "none"),
        ("paged", 12, "paged"),
        ("quantized", 12, "quantized"),
    ],
)
def test_result_keys_and_legacy_metadata(
    *, cache_type: str, heads: int, key: str
) -> None:
    baseline = benchmark.CacheBenchmark.model_validate(
        {
            "cache_type": cache_type,
            "num_kv_heads": heads,
            "gpu_type": "cpu",
            "points": [],
        }
    )
    assert baseline.attention_backend == "expanded"
    assert baseline.dtype == "float32"
    assert benchmark._result_key(baseline) == key
    if cache_type in ("basic", "gqa", "none"):
        grouped = baseline.model_copy(update={"attention_backend": "grouped"})
        assert benchmark._result_key(grouped) == f"{key}_grouped_float32"
