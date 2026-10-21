"""Offline checks for paired baseline runs, report safety, and provenance."""

import hashlib
import inspect
import json
import subprocess
import sys
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import benchmark
import benchmark_baseline as sweep
from model import ModelConfig, TinyTransformer


@pytest.fixture
def modal_calls(*, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    report = {
        variant: benchmark.CacheBenchmark(
            cache_type=variant, gpu_type="offline", points=[]
        ).model_dump(mode="json")
        for variant in sweep.VARIANTS
    }
    report["metadata"] = {}
    calls = SimpleNamespace(
        output=Mock(return_value=nullcontext()),
        run=Mock(return_value=nullcontext()),
        remote=Mock(return_value=report),
    )
    monkeypatch.setattr(sweep.modal, "enable_output", calls.output)
    monkeypatch.setattr(sweep.app, "run", calls.run)
    monkeypatch.setattr(sweep.experiment, "remote", calls.remote)
    return calls


def test_parse_args_defaults(*, tmp_path: Path) -> None:
    args = sweep.parse_args(argv=["--output", str(tmp_path / "report.json")])
    assert args.points == [
        (64, 1),
        (128, 1),
        (256, 1),
        (512, 1),
        (1000, 1),
        (256, 8),
        (256, 16),
        (256, 64),
        (256, 128),
    ]
    assert args.trials == 10
    assert args.steps == 20
    assert args.seed == 42


def test_parse_args_exact_points_and_capacity(*, tmp_path: Path) -> None:
    args = sweep.parse_args(
        argv=[
            "--output",
            str(tmp_path / "report.json"),
            "--points",
            "256:8",
            "1003:1",
            "64:16",
            "--trials",
            "2",
            "--steps",
            "20",
            "--seed",
            "17",
        ]
    )
    assert args.points == [(256, 8), (1003, 1), (64, 16)]
    assert max(context for context, _ in args.points) + 1 + args.steps == 1024
    assert args.trials == 2
    assert args.seed == 17


@pytest.mark.parametrize(
    "options",
    [
        ["--points"],
        ["--points", "256"],
        ["--points", "1:2:3"],
        ["--points", "x:1"],
        ["--points", "1:"],
        ["--points", "0:1"],
        ["--points", "64:0"],
        ["--points=-1:1"],
        ["--points", "64:-1"],
        ["--points", "1004:1"],
        ["--trials", "0"],
        ["--trials", "-1"],
        ["--steps", "0"],
        ["--steps", "-1"],
        ["--steps", "24"],
        ["--overwrite"],
    ],
)
def test_invalid_args_never_launch(
    *, options: list[str], tmp_path: Path, modal_calls: SimpleNamespace
) -> None:
    with pytest.raises(SystemExit, match="2"):
        sweep.main(argv=["--output", str(tmp_path / "report.json"), *options])
    modal_calls.output.assert_not_called()
    modal_calls.run.assert_not_called()
    modal_calls.remote.assert_not_called()


def test_output_is_required(*, modal_calls: SimpleNamespace) -> None:
    with pytest.raises(SystemExit, match="2"):
        sweep.main(argv=[])
    modal_calls.run.assert_not_called()
    modal_calls.remote.assert_not_called()


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
def test_existing_output_never_launches(
    *, tmp_path: Path, modal_calls: SimpleNamespace, kind: str
) -> None:
    output = tmp_path / "historic.json"
    target = tmp_path / "target.json"
    original = b"unverifiable historic report, preserve byte for byte\n"
    if kind == "file":
        output.write_bytes(original)
    elif kind == "directory":
        output.mkdir()
    else:
        if kind == "symlink":
            target.write_bytes(original)
        output.symlink_to(target)
    with pytest.raises(SystemExit, match="2"):
        sweep.main(argv=["--output", str(output)])
    if kind in ("file", "symlink"):
        assert output.read_bytes() == original
    elif kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink() and not target.exists()
    modal_calls.output.assert_not_called()
    modal_calls.run.assert_not_called()
    modal_calls.remote.assert_not_called()


@pytest.mark.parametrize("symlink", [False, True])
def test_output_created_during_job_is_not_overwritten(
    *, tmp_path: Path, modal_calls: SimpleNamespace, symlink: bool
) -> None:
    output = tmp_path / "report.json"
    target = tmp_path / "target.json"
    original = b"report created while the benchmark ran\n"
    report = modal_calls.remote.return_value

    def finish_job(**kwargs: object) -> dict:
        if symlink:
            output.symlink_to(target)
        else:
            output.write_bytes(original)
        return report

    modal_calls.remote.side_effect = finish_job
    with pytest.raises(FileExistsError):
        sweep.main(argv=["--output", str(output)])
    if symlink:
        assert output.is_symlink() and not target.exists()
    else:
        assert output.read_bytes() == original
    modal_calls.remote.assert_called_once()


def test_capture_sources_uses_imported_files_not_cwd(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("benchmark.py", "generate.py", "model.py", "shared.py", "uv.lock"):
        (tmp_path / name).write_text("unrelated working-directory file")
    monkeypatch.chdir(tmp_path)
    snapshot = sweep.capture_sources()
    assert datetime.fromisoformat(snapshot["captured_at_utc"]).tzinfo == UTC
    assert set(snapshot["files"]) == {
        "benchmark_baseline.py",
        "benchmark.py",
        "generate.py",
        "model.py",
        "shared.py",
        "pyproject.toml",
        "uv.lock",
        "tests/conftest.py",
        "tests/test_baseline_sweep.py",
        *(f"tests/{name}" for name in sweep.PREFLIGHT_TESTS),
    }
    for name in ("benchmark_baseline", "benchmark", "generate", "model", "shared"):
        assert snapshot["files"][f"{name}.py"]["path"] == str(
            Path(inspect.getfile(sys.modules[name])).resolve()
        )
    for source in snapshot["files"].values():
        content = Path(source["path"]).read_bytes()
        assert source["content"].encode("utf-8") == content
        assert source["sha256"] == hashlib.sha256(content).hexdigest()


def test_main_captures_before_launch_and_writes_compatible_entries(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modal_calls: SimpleNamespace
) -> None:
    model_path = Path(inspect.getfile(sys.modules["model"])).resolve()
    original = b"# model source at launch\r\n"
    current = original
    read_bytes = Path.read_bytes

    def read_source(path: Path) -> bytes:
        return current if path == model_path else read_bytes(path)

    def enable_output():
        nonlocal current
        current = b"# model source edited after the snapshot\n"
        return nullcontext()

    monkeypatch.setattr(Path, "read_bytes", read_source)
    modal_calls.output.side_effect = enable_output
    output = tmp_path / "nested" / "report.json"
    sweep.main(
        argv=[
            "--output",
            str(output),
            "--points",
            "256:8",
            "64:1",
            "--trials",
            "2",
            "--steps",
            "3",
            "--seed",
            "17",
        ]
    )
    saved = json.loads(output.read_text())
    assert set(saved) == {"basic", "none", "metadata", "source_snapshot"}
    for variant in sweep.VARIANTS:
        entry = benchmark.CacheBenchmark.model_validate(saved[variant])
        assert entry.cache_type == variant
    files = saved["source_snapshot"]["files"]
    assert files["model.py"]["content"].encode("utf-8") == original
    assert files["model.py"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert saved["metadata"]["source_sha256"] == {
        name: source["sha256"] for name, source in files.items()
    }
    modal_calls.run.assert_called_once_with()
    modal_calls.remote.assert_called_once_with(
        points=[(256, 8), (64, 1)], trials=2, steps=3, seed=17
    )


@pytest.fixture
def offline_experiment(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    weights_path = tmp_path / "model.safetensors"
    weights_path.write_bytes(b"offline checkpoint bytes\r\n")
    model = Mock()
    model.to.return_value = model
    events = []

    def seed_cpu(*, seed: int) -> None:
        events.append(("seed", seed))
        torch.random.default_generator.manual_seed(seed)

    calls = SimpleNamespace(
        preflight=Mock(
            return_value=subprocess.CompletedProcess(
                args=["pytest"], returncode=0, stdout="correctness passed\n", stderr=""
            )
        ),
        download=Mock(return_value=str(weights_path)),
        load_file=Mock(return_value={"offline": torch.tensor(1.0)}),
        load_weights=Mock(),
        constructor=Mock(return_value=model),
        threads=Mock(),
        seed=Mock(side_effect=seed_cpu),
        model=model,
        events=events,
        weights_path=weights_path,
        tokenizer=SimpleNamespace(n_vocab=23, eot_token=22),
    )
    monkeypatch.setattr(sweep.subprocess, "run", calls.preflight)
    monkeypatch.setattr(sweep, "hf_hub_download", calls.download)
    monkeypatch.setattr(sweep, "load_file", calls.load_file)
    monkeypatch.setattr(sweep, "load_gpt2_weights", calls.load_weights)
    monkeypatch.setattr(sweep, "TinyTransformer", calls.constructor)
    monkeypatch.setattr(sweep, "seed_everything", calls.seed)
    monkeypatch.setattr(sweep.torch, "set_num_threads", calls.threads)
    monkeypatch.setattr(sweep.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(
        sweep.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="offline A100", uuid="GPU-offline"),
    )
    monkeypatch.setattr(sweep.tiktoken, "get_encoding", lambda _: calls.tokenizer)
    monkeypatch.setattr(sweep, "GPT2_SMALL_D_MODEL", 24)
    monkeypatch.setattr(sweep, "GPT2_SMALL_N_LAYERS", 1)
    return calls


def test_failed_preflight_stops_before_loading(
    *, offline_experiment: SimpleNamespace
) -> None:
    calls = offline_experiment
    calls.preflight.return_value = subprocess.CompletedProcess(
        args=["pytest"], returncode=1, stdout="correctness failed", stderr="details"
    )
    with pytest.raises(RuntimeError, match="correctness faileddetails"):
        sweep.experiment.local(points=[(4, 1)], trials=1, steps=1, seed=42)
    calls.preflight.assert_called_once()
    assert calls.preflight.call_args.args[0] == [
        sys.executable,
        "-m",
        "pytest",
        *(f"/tests/{name}" for name in sweep.PREFLIGHT_TESTS),
        "-q",
    ]
    assert calls.preflight.call_args.kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert calls.preflight.call_args.kwargs["env"]["MKL_NUM_THREADS"] == "1"
    calls.download.assert_not_called()
    calls.constructor.assert_not_called()


@pytest.mark.parametrize(
    "points,trials,steps",
    [([], 1, 1), ([(0, 1)], 1, 1), ([(4, 1)], 0, 1), ([(1004, 1)], 1, 20)],
)
def test_remote_revalidates_before_preflight(
    *,
    offline_experiment: SimpleNamespace,
    points: list[tuple[int, int]],
    trials: int,
    steps: int,
) -> None:
    with pytest.raises(ValueError):
        sweep.experiment.local(points=points, trials=trials, steps=steps, seed=42)
    offline_experiment.preflight.assert_not_called()
    offline_experiment.download.assert_not_called()


def test_one_model_alternating_variants_reseeding_and_metadata(
    *, offline_experiment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = offline_experiment
    measured = []
    points = [(4, 1), (5, 2), (6, 1)]

    def measure(**kwargs) -> benchmark.BenchmarkPoint:
        calls.events.append(("point", kwargs["cache_type"]))
        prompts = [
            torch.randint(
                0,
                kwargs["config"].vocab_size,
                (kwargs["batch_size"], kwargs["context_length"]),
            )
            for _ in range(kwargs["num_trials"])
        ]
        result = benchmark.BenchmarkPoint(
            context_length=kwargs["context_length"],
            batch_size=kwargs["batch_size"],
            decode_latency=None,
            throughput_tokens_per_sec=None,
            num_trials=0,
            cache_bytes_used=0,
            peak_memory_bytes=123,
            oom=True,
        )
        measured.append((kwargs, prompts, result))
        return result

    monkeypatch.setattr(benchmark, "_run_sweep_point", measure)
    report = sweep.experiment.local(points=points, trials=2, steps=3, seed=17)
    expected_variants = ["basic", "none", "none", "basic", "basic", "none"]
    assert calls.events == [("seed", 17)] + [
        event
        for variant in expected_variants
        for event in (("seed", 17), ("point", variant))
    ]
    calls.constructor.assert_called_once()
    config = calls.constructor.call_args.kwargs["config"]
    assert config.max_seq_len == 1024
    assert config.num_kv_heads == config.num_heads == 12
    assert config.attention_backend == "grouped"
    calls.model.to.assert_called_once_with(
        device=torch.device("cuda"), dtype=torch.float32
    )
    calls.model.eval.assert_called_once_with()
    calls.download.assert_called_once_with(repo_id="gpt2", filename="model.safetensors")
    calls.load_file.assert_called_once_with(str(calls.weights_path), device="cuda")
    calls.load_weights.assert_called_once_with(
        calls.model, gpt2_state_dict=calls.load_file.return_value
    )
    calls.threads.assert_called_once_with(1)
    for index, (kwargs, _, result) in enumerate(measured):
        assert kwargs["model"] is calls.model
        assert kwargs["config"] is config
        assert kwargs["device"] == torch.device("cuda")
        assert kwargs["cache_type"] == expected_variants[index]
        assert (kwargs["context_length"], kwargs["batch_size"]) == points[index // 2]
        assert kwargs["num_trials"] == 2
        assert kwargs["max_new_tokens"] == 3
        assert kwargs["eos_token_id"] == calls.tokenizer.eot_token
        assert report[kwargs["cache_type"]]["points"][index // 2] == result.model_dump()
    for index in range(0, len(measured), 2):
        for first, second in zip(
            measured[index][1], measured[index + 1][1], strict=True
        ):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    for variant in sweep.VARIANTS:
        entry = benchmark.CacheBenchmark.model_validate(report[variant])
        assert entry.seed == 17 and entry.max_new_tokens == 3
        assert entry.max_seq_len == 1024 and entry.dtype == "float32"
        assert entry.attention_backend == "grouped" and entry.num_kv_heads == 12
        assert [(p.context_length, p.batch_size) for p in entry.points] == points
    metadata = report["metadata"]
    assert metadata["gpu"] == "offline A100"
    assert metadata["gpu_uuid"] == "GPU-offline"
    assert metadata["torch"] == torch.__version__
    assert metadata["cuda"] == torch.version.cuda
    assert metadata["cpu_threads"] == 1
    assert (metadata["trials"], metadata["steps"], metadata["seed"]) == (2, 3, 17)
    assert metadata["warmup_steps"] == 1 and not metadata["forced_tokens"]
    assert metadata["single_container"]
    assert metadata["tests"] == "correctness passed\n"
    assert metadata["checkpoint_sha256"] == hashlib.sha256(
        calls.weights_path.read_bytes()
    ).hexdigest()
    assert metadata["execution_order"] == [
        {"context_length": context, "batch_size": batch, "cache_type": variant}
        for (context, batch), variant in zip(
            [point for point in points for _ in range(2)],
            expected_variants,
            strict=True,
        )
    ]
    assert datetime.fromisoformat(metadata["started_at_utc"]) <= datetime.fromisoformat(
        metadata["finished_at_utc"]
    )
    assert metadata["started_at_utc"].endswith("+00:00")


@pytest.mark.parametrize("eos_token_id", [0, -1], ids=["early_eos", "all_steps"])
def test_real_sweep_preserves_eos_steps_and_matching_trial_prompts(
    *,
    offline_experiment: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    eos_token_id: int,
) -> None:
    calls = offline_experiment
    calls.tokenizer.eot_token = eos_token_id
    config = ModelConfig(
        vocab_size=23,
        d_model=24,
        n_layers=1,
        num_heads=12,
        num_kv_heads=12,
        attention_backend="grouped",
        max_seq_len=1024,
    )
    model = TinyTransformer(config=config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    monkeypatch.setattr(model, "to", Mock(return_value=model))
    calls.constructor.return_value = model
    prompts = []

    def record_prompt(module, args, kwargs) -> None:
        if args[0].shape[1] == 4:
            prompts.append(args[0].clone())

    model.register_forward_pre_hook(record_prompt, with_kwargs=True)
    run_point = benchmark._run_sweep_point

    def cpu_point(**kwargs) -> benchmark.BenchmarkPoint:
        assert kwargs["model"] is model
        assert kwargs["config"] == config
        return run_point(**{**kwargs, "device": torch.device("cpu")})

    monkeypatch.setattr(benchmark, "_run_sweep_point", cpu_point)
    report = sweep.experiment.local(points=[(4, 2)], trials=2, steps=2, seed=17)
    assert len(prompts) == 4
    assert not torch.equal(prompts[0], prompts[1])
    torch.testing.assert_close(prompts[0], prompts[2], rtol=0, atol=0)
    torch.testing.assert_close(prompts[1], prompts[3], rtol=0, atol=0)
    for variant in sweep.VARIANTS:
        point = benchmark.CacheBenchmark.model_validate(report[variant]).points[0]
        assert point.num_trials == 2 and not point.oom
        assert point.timed_steps == (0 if eos_token_id == 0 else 4)
        assert (point.decode_latency is None) == (eos_token_id == 0)
        bytes_per_position = 2 * 2 * 24 * 4 if variant == "basic" else 0
        assert point.cache_bytes_used == 4 * bytes_per_position
        assert point.cache_bytes_allocated == 1024 * bytes_per_position
