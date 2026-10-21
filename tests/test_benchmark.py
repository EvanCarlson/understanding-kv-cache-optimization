"""Regress prefill cache accounting and non-destructive benchmark report writes."""

import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import benchmark
import generate
from model import BlockedKVCache, KVCacheProtocol, ModelConfig, TinyTransformer


@dataclass(frozen=True)
class CacheSnapshot:
    length: int
    used: int
    allocated: int


@pytest.fixture(params=[1, 2], ids=["one_layer", "two_layers"])
def config(*, request: pytest.FixtureRequest) -> ModelConfig:
    return ModelConfig(
        vocab_size=23,
        d_model=24,
        max_seq_len=140,
        n_layers=request.param,
        num_heads=12,
        num_kv_heads=12,
    )


@pytest.mark.parametrize("context_length", [127, 128])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_prefill_and_warmup_snapshots_paged_bytes(
    *, config: ModelConfig, context_length: int, batch_size: int
) -> None:
    model = TinyTransformer(config=config).eval()
    prefill = benchmark._prefill_and_warmup(
        model=model,
        cache_type="paged",
        config=config,
        context_length=context_length,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    cache = prefill.cache
    assert isinstance(cache, BlockedKVCache)
    assert cache.block_size == 128
    bytes_per_token = 2 * config.n_layers * batch_size * config.d_model * 4
    assert prefill.cache_bytes_allocated == 128 * bytes_per_token
    assert prefill.cache_bytes_used == context_length * bytes_per_token
    assert cache.length == context_length + 1
    assert cache.used_nbytes == (context_length + 1) * bytes_per_token
    warmup_capacity = 128 if context_length == 127 else 256
    assert cache.allocated_nbytes == warmup_capacity * bytes_per_token


def test_run_sweep_point_keeps_one_latency_list_per_trial(
    *, config: ModelConfig
) -> None:
    model = TinyTransformer(config=config).eval()

    point = benchmark._run_sweep_point(
        model=model,
        cache_type="basic",
        config=config,
        context_length=8,
        batch_size=1,
        num_trials=3,
        max_new_tokens=2,
        eos_token_id=-1,
        device=torch.device("cpu"),
    )

    assert point.num_trials == 3
    assert [len(trial) for trial in point.trial_latencies_ms] == [2, 2, 2]
    assert point.timed_steps == 6


@pytest.mark.parametrize("context_length", [127, 128])
def test_run_sweep_point_paged_decode_growth_preserves_prefill_bytes(
    *, config: ModelConfig, context_length: int
) -> None:
    model = TinyTransformer(config=config).eval()
    snapshots: list[CacheSnapshot] = []

    def record_cache(
        module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, KVCacheProtocol],
        output: torch.Tensor,
    ) -> None:
        cache = kwargs["cache"]
        snapshots.append(
            CacheSnapshot(
                length=cache.length,
                used=cache.used_nbytes,
                allocated=cache.allocated_nbytes,
            )
        )

    hook = model.register_forward_hook(record_cache, with_kwargs=True)
    try:
        point = benchmark._run_sweep_point(
            model=model,
            cache_type="paged",
            config=config,
            context_length=context_length,
            batch_size=2,
            num_trials=2,
            max_new_tokens=2,
            eos_token_id=-1,
            device=torch.device("cpu"),
        )
    finally:
        hook.remove()

    bytes_per_token = 2 * config.n_layers * 2 * config.d_model * 4
    assert len(snapshots) == 8
    for start in (0, 4):
        prefill, warmup, first_decode, second_decode = snapshots[start : start + 4]
        assert prefill.length == context_length
        assert prefill.allocated == 128 * bytes_per_token
        assert prefill.used == context_length * bytes_per_token
        assert warmup.length == context_length + 1
        assert first_decode.length == context_length + 2
        assert second_decode.length == context_length + 3
        assert second_decode.allocated == 256 * bytes_per_token
        assert point.cache_bytes_used == prefill.used
        assert point.cache_bytes_allocated == prefill.allocated
    assert point.num_trials == 2
    assert point.timed_steps == 4
    assert not point.oom
    assert point.decode_latency is not None
    assert point.throughput_tokens_per_sec == pytest.approx(
        2 / (point.decode_latency.p50_ms / 1000)
    )


@pytest.mark.parametrize("context_length", [127, 128])
def test_run_trial_eos_preserves_prefill_bytes(
    *, config: ModelConfig, context_length: int
) -> None:
    model = TinyTransformer(config=config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    latencies, used, allocated = benchmark._run_trial(
        model=model,
        cache_type="paged",
        config=config,
        context_length=context_length,
        batch_size=1,
        max_new_tokens=2,
        eos_token_id=0,
        device=torch.device("cpu"),
    )
    bytes_per_token = 2 * config.n_layers * config.d_model * 4
    assert latencies == []
    assert used == context_length * bytes_per_token
    assert allocated == 128 * bytes_per_token


@pytest.fixture
def legacy_path(*, tmp_path: Path) -> Path:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "basic": {
                    "cache_type": "basic",
                    "gpu_type": "A100",
                    "points": [
                        {
                            "context_length": 64,
                            "batch_size": 1,
                            "decode_latency": {
                                "p50_ms": 1.0,
                                "p95_ms": 2.0,
                                "p99_ms": 3.0,
                            },
                            "throughput_tokens_per_sec": 1000.0,
                            "num_trials": 10,
                            "cache_bytes_used": 1234,
                            "peak_memory_bytes": 5678,
                            "oom": False,
                            "historical_note": "keep point metadata",
                        }
                    ],
                    "historical_note": "keep report metadata",
                },
                "none": {"cache_type": "none", "gpu_type": "A100", "points": []},
                "gqa_h3_sdpa_bfloat16": {
                    "cache_type": "gqa",
                    "num_kv_heads": 3,
                    "gpu_type": "A100",
                    "points": [],
                    "attention_backend": "sdpa",
                    "dtype": "bfloat16",
                },
            }
        )
    )
    return path


def test_load_results_reads_historical_schemas(*, legacy_path: Path) -> None:
    original = legacy_path.read_bytes()
    results = benchmark._load_results(legacy_path)
    assert results["basic"].dtype == "float32"
    assert results["basic"].attention_backend == "expanded"
    assert results["basic"].points[0].cache_bytes_allocated == 0
    assert results["basic"].points[0].timed_steps == 0
    historical = results["gqa_h3_sdpa_bfloat16"]
    assert historical.dtype == "bfloat16"
    assert historical.attention_backend == "sdpa"
    assert benchmark._result_key(historical) == "gqa_h3_sdpa_bfloat16"
    assert legacy_path.read_bytes() == original


def test_save_results_creates_new_report(*, tmp_path: Path) -> None:
    path = tmp_path / "new" / "report.json"
    result = benchmark.CacheBenchmark(cache_type="basic", gpu_type="cpu", points=[])
    benchmark._save_results(path, {"basic": result})
    assert json.loads(path.read_text()) == {"basic": result.model_dump()}
    assert benchmark._load_results(path) == {"basic": result}


def test_save_results_rejects_conflict_without_partial_write(
    *, legacy_path: Path
) -> None:
    original = legacy_path.read_bytes()
    result = benchmark.CacheBenchmark(cache_type="basic", gpu_type="cpu", points=[])
    gqa = result.model_copy(update={"cache_type": "gqa", "num_kv_heads": 1})
    with pytest.raises(FileExistsError, match="basic.*--overwrite"):
        benchmark._save_results(legacy_path, {"gqa_h1": gqa, "basic": result})
    assert legacy_path.read_bytes() == original


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_results_preserves_unrelated_raw_entries(
    *, legacy_path: Path, overwrite: bool
) -> None:
    original = json.loads(legacy_path.read_text())
    result = benchmark.CacheBenchmark(
        cache_type="basic" if overwrite else "gqa",
        num_kv_heads=12 if overwrite else 1,
        gpu_type="cpu",
        points=[],
        seed=7,
    )
    key = benchmark._result_key(result)
    benchmark._save_results(legacy_path, {key: result}, overwrite=overwrite)
    expected = {**original, key: result.model_dump()}
    assert json.loads(legacy_path.read_text()) == expected


def test_parse_args_overwrite_is_explicit() -> None:
    assert not benchmark.parse_args([]).overwrite
    assert benchmark.parse_args(["--overwrite"]).overwrite


@pytest.mark.parametrize("modal", [False, True])
@pytest.mark.parametrize(
    "cache_type,heads,backend",
    [
        ("basic", 12, "expanded"),
        ("none", 12, "expanded"),
        ("paged", 12, "expanded"),
        ("quantized", 12, "expanded"),
        ("gqa", 3, "expanded"),
        ("gqa", 1, "expanded"),
        ("gqa", 3, "grouped"),
        ("basic", 12, "grouped"),
        ("none", 12, "grouped"),
    ],
)
def test_main_rejects_conflicts_before_loading_or_remote_launch(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modal: bool,
    cache_type: str,
    heads: int,
    backend: str,
) -> None:
    path = tmp_path / "report.json"
    result = benchmark.CacheBenchmark(
        cache_type=cache_type,
        num_kv_heads=heads,
        attention_backend=backend,
        gpu_type="cpu",
        points=[],
    )
    key = benchmark._result_key(result)
    original = json.dumps({key: result.model_dump()})
    path.write_text(original)
    forbidden = Mock(side_effect=AssertionError("Must reject before execution"))
    monkeypatch.setattr(generate, "hf_hub_download", forbidden)
    monkeypatch.setattr(benchmark.tiktoken, "get_encoding", forbidden)
    monkeypatch.setattr(benchmark.modal, "enable_output", forbidden)
    monkeypatch.setattr(benchmark.app, "run", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--output",
            str(path),
            "--cache-type",
            cache_type,
            "--num-kv-heads",
            str(heads),
            "--attention-backend",
            backend,
            *(["--modal"] if modal else []),
        ],
    )
    with pytest.raises(FileExistsError, match="--overwrite") as error:
        benchmark.main()
    assert key in str(error.value)
    forbidden.assert_not_called()
    assert path.read_text() == original


@pytest.fixture
def remote_call(*, monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setattr(benchmark.modal, "enable_output", nullcontext)
    monkeypatch.setattr(benchmark.app, "run", nullcontext)
    remote = Mock()
    endpoint = Mock()
    endpoint.with_options.return_value.remote = remote
    monkeypatch.setattr(benchmark, "remote_run_benchmark", endpoint)
    return remote


def test_main_remote_forwards_retained_options(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_call: Mock
) -> None:
    path = tmp_path / "report.json"
    result = benchmark.CacheBenchmark(
        cache_type="gqa",
        num_kv_heads=1,
        gpu_type="A100",
        points=[],
        attention_backend="grouped",
        seed=7,
    )
    remote_call.return_value = (result.model_dump(), None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--modal",
            "--no-trace",
            "--output",
            str(path),
            "--cache-type",
            "gqa",
            "--num-kv-heads",
            "1",
            "--attention-backend",
            "grouped",
            "--seed",
            "7",
        ],
    )
    benchmark.main()
    remote_call.assert_called_once_with(
        cache_type="gqa",
        context_lengths=benchmark.DEFAULT_CONTEXT_LENGTHS,
        batch_sizes=benchmark.DEFAULT_BATCH_SIZES,
        num_trials=benchmark.DEFAULT_NUM_TRIALS,
        max_new_tokens=benchmark.DEFAULT_MAX_NEW_TOKENS,
        model_repo=benchmark.DEFAULT_MODEL_REPO,
        gpu_type="A100",
        capture_trace=False,
        seed=7,
        num_kv_heads=1,
        attention_backend="grouped",
    )
    assert json.loads(path.read_text()) == {
        "gqa_h1_grouped_float32": result.model_dump()
    }


def test_main_explicit_overwrite_preserves_other_entries(
    *, legacy_path: Path, monkeypatch: pytest.MonkeyPatch, remote_call: Mock
) -> None:
    original = json.loads(legacy_path.read_text())
    result = benchmark.CacheBenchmark(
        cache_type="basic", gpu_type="A100", points=[], seed=7
    )
    remote_call.return_value = (result.model_dump(), None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--modal",
            "--no-trace",
            "--overwrite",
            "--output",
            str(legacy_path),
        ],
    )
    benchmark.main()
    remote_call.assert_called_once()
    assert json.loads(legacy_path.read_text()) == {
        **original,
        "basic": result.model_dump(),
    }


@pytest.mark.parametrize("overwrite", [False, True])
def test_main_rechecks_conflicts_and_merges_entries_added_during_run(
    *,
    legacy_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_call: Mock,
    overwrite: bool,
) -> None:
    original = json.loads(legacy_path.read_text())
    result = benchmark.CacheBenchmark(
        cache_type="gqa", num_kv_heads=3, gpu_type="A100", points=[], seed=7
    )
    late_entries = {
        **original,
        "paged": {"cache_type": "paged", "gpu_type": "cpu", "points": []},
        "gqa_h3": result.model_copy(update={"seed": 99}).model_dump(),
    }
    late_payload = json.dumps(late_entries)

    def remote_run(**kwargs: object) -> tuple[dict, bytes]:
        legacy_path.write_text(late_payload)
        return result.model_dump(), b"trace"

    remote_call.side_effect = remote_run
    traces = tmp_path / "traces"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--modal",
            "--cache-type",
            "gqa",
            "--output",
            str(legacy_path),
            "--traces-dir",
            str(traces),
            *(["--overwrite"] if overwrite else []),
        ],
    )
    if overwrite:
        benchmark.main()
        assert json.loads(legacy_path.read_text()) == {
            **late_entries,
            "gqa_h3": result.model_dump(),
        }
        assert (traces / "gqa_h3.json").read_bytes() == b"trace"
    else:
        with pytest.raises(FileExistsError, match="gqa_h3.*--overwrite"):
            benchmark.main()
        assert legacy_path.read_text() == late_payload
        assert not traces.exists()
    remote_call.assert_called_once()
