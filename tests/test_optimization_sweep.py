"""Offline checks for the FP32 sweep, report safety, and source provenance."""

import hashlib
import inspect
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import benchmark_optimizations as sweep
from model import KVCacheProtocol, ModelConfig, TinyTransformer


@pytest.fixture
def report() -> sweep.SweepReport:
    return sweep.SweepReport(
        metadata=sweep.SweepMetadata(
            gpu="offline",
            gpu_uuid="offline",
            torch=torch.__version__,
            cuda=None,
            cpu_threads=1,
            rounds=2,
            trials=10,
            steps=20,
            seed=42,
            stages=list(sweep.STAGES),
            points=[(1000, 1)],
            tests="offline test result",
            checkpoint_sha256="checkpoint-hash",
        )
    )


@dataclass(frozen=True)
class ModalCalls:
    output: Mock
    run: Mock
    remote: Mock


@pytest.fixture
def modal_calls(
    *, monkeypatch: pytest.MonkeyPatch, report: sweep.SweepReport
) -> ModalCalls:
    calls = ModalCalls(
        output=Mock(return_value=nullcontext()),
        run=Mock(return_value=nullcontext()),
        remote=Mock(return_value=report.model_dump(mode="json")),
    )
    monkeypatch.setattr(sweep.modal, "enable_output", calls.output)
    monkeypatch.setattr(sweep.app, "run", calls.run)
    monkeypatch.setattr(sweep.experiment, "remote", calls.remote)
    return calls


def test_parse_args_fp32_defaults(*, tmp_path: Path) -> None:
    args = sweep.parse_args(argv=["--output", str(tmp_path / "report.json")])
    assert args.points == sweep.DEFAULT_POINTS
    assert args.stages == ["expanded_fp32", "grouped_fp32"]
    assert sweep.STAGES == {
        "expanded_fp32": "expanded",
        "grouped_fp32": "grouped",
        "repeat_kv_fp32": "repeat_kv",
    }
    assert args.seed == 42
    assert not args.overwrite


def test_parse_args_post_command(*, tmp_path: Path) -> None:
    args = sweep.parse_args(
        argv=[
            "--stages",
            "expanded_fp32",
            "grouped_fp32",
            "--context-lengths",
            "64",
            "128",
            "256",
            "512",
            "1000",
            "--batch-sizes",
            "1",
            "8",
            "16",
            "64",
            "128",
            "--rounds",
            "2",
            "--trials",
            "10",
            "--steps",
            "20",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    assert args.points == [
        (context, batch)
        for context in [64, 128, 256, 512, 1000]
        for batch in [1, 8, 16, 64, 128]
    ]
    assert args.trials == 10
    assert args.rounds == 2
    assert args.steps == 20


@pytest.mark.parametrize(
    "options",
    [
        ["--context-lengths", "256"],
        ["--batch-sizes", "1", "8"],
        ["--context-lengths", "0", "--batch-sizes", "1"],
        ["--context-lengths", "256", "--batch-sizes", "-1"],
        ["--context-lengths", "1024", "--batch-sizes", "1"],
        ["--rounds", "0"],
        ["--trials", "-1"],
        ["--steps", "0"],
        ["--steps", "24"],
        ["--profile"],
        ["--compile"],
        ["--dtype", "bfloat16"],
        ["--stages", "flash_bf16"],
        ["--stages", "static_grouped_bf16"],
        ["--stages", "graph_grouped_bf16"],
    ],
)
def test_parse_args_rejects_invalid_sweep(
    *, options: list[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="2"):
        sweep.parse_args(argv=["--output", str(tmp_path / "report.json"), *options])


@pytest.mark.parametrize(
    "options",
    [
        ["--steps", "23"],
        ["--context-lengths", "1003", "--batch-sizes", "1"],
        ["--context-lengths", "2", "--batch-sizes", "1", "--steps", "1021"],
    ],
)
def test_parse_args_accepts_exact_capacity(
    *, options: list[str], tmp_path: Path
) -> None:
    args = sweep.parse_args(argv=["--output", str(tmp_path / "report.json"), *options])
    assert max(context for context, _ in args.points) + 1 + args.steps == 1024


def test_main_existing_report_never_launches(
    *, tmp_path: Path, modal_calls: ModalCalls, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "historic.json"
    original = b'{"metadata":{"source_sha256":{"model.py":"historic-hash"}}}\n'
    output.write_bytes(original)
    with pytest.raises(SystemExit, match="2"):
        sweep.main(argv=["--output", str(output)])
    assert "--overwrite" in capsys.readouterr().err
    assert output.read_bytes() == original
    modal_calls.output.assert_not_called()
    modal_calls.run.assert_not_called()
    modal_calls.remote.assert_not_called()


@pytest.mark.parametrize("target_exists", [False, True])
def test_main_existing_symlink_never_launches(
    *, tmp_path: Path, modal_calls: ModalCalls, target_exists: bool
) -> None:
    target = tmp_path / "target.json"
    if target_exists:
        target.write_text("historic report")
    output = tmp_path / "report.json"
    output.symlink_to(target)
    with pytest.raises(SystemExit, match="2"):
        sweep.main(argv=["--output", str(output)])
    assert output.is_symlink()
    assert target.exists() == target_exists
    modal_calls.run.assert_not_called()
    modal_calls.remote.assert_not_called()


def test_main_explicit_overwrite(*, tmp_path: Path, modal_calls: ModalCalls) -> None:
    output = tmp_path / "report.json"
    output.write_text("old report")
    sweep.main(argv=["--output", str(output), "--overwrite"])
    saved = sweep.SweepReport.model_validate_json(output.read_text())
    assert saved.source_snapshot is not None
    assert saved.metadata.dtype == "float32"
    modal_calls.run.assert_called_once_with(detach=True)
    modal_calls.remote.assert_called_once()


@pytest.mark.parametrize("overwrite", [False, True])
def test_main_output_created_during_job(
    *,
    tmp_path: Path,
    modal_calls: ModalCalls,
    report: sweep.SweepReport,
    overwrite: bool,
) -> None:
    output = tmp_path / "report.json"
    competing_report = b"a report created while this job was running\n"

    def finish_job(**kwargs) -> dict:
        output.write_bytes(competing_report)
        return report.model_dump(mode="json")

    modal_calls.remote.side_effect = finish_job
    argv = ["--output", str(output)]
    if overwrite:
        sweep.main(argv=[*argv, "--overwrite"])
        assert sweep.SweepReport.model_validate_json(output.read_text()).source_snapshot
    else:
        with pytest.raises(FileExistsError):
            sweep.main(argv=argv)
        assert output.read_bytes() == competing_report
    modal_calls.run.assert_called_once_with(detach=True)
    modal_calls.remote.assert_called_once()


def test_capture_sources_uses_imported_files_not_cwd(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("benchmark.py", "generate.py", "model.py", "shared.py", "uv.lock"):
        (tmp_path / name).write_text("unrelated working-directory file")
    monkeypatch.chdir(tmp_path)
    snapshot = sweep.capture_sources()
    assert set(snapshot.files) == {
        "benchmark_optimizations.py",
        "benchmark.py",
        "generate.py",
        "model.py",
        "shared.py",
        "pyproject.toml",
        "uv.lock",
        "tests/conftest.py",
        *(f"tests/{name}" for name in sweep.PREFLIGHT_TESTS),
    }
    for name in ("benchmark_optimizations", "benchmark", "generate", "model", "shared"):
        assert snapshot.files[f"{name}.py"].path == str(
            Path(inspect.getfile(sys.modules[name])).resolve()
        )
    for source in snapshot.files.values():
        content = Path(source.path).read_bytes()
        assert source.content.encode("utf-8") == content
        assert source.sha256 == hashlib.sha256(content).hexdigest()


def test_main_snapshot_precedes_app_launch_and_survives_source_edits(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, modal_calls: ModalCalls
) -> None:
    model_path = Path(inspect.getfile(sys.modules["model"])).resolve()
    original = b"# source at launch\r\n"
    current = original
    read_bytes = Path.read_bytes

    def read_source(path: Path) -> bytes:
        return current if path == model_path else read_bytes(path)

    def open_app():
        nonlocal current
        current = b"# edited after the snapshot\n"
        return nullcontext()

    monkeypatch.setattr(Path, "read_bytes", read_source)
    modal_calls.output.side_effect = open_app
    output = tmp_path / "nested" / "report.json"
    sweep.main(
        argv=["--output", str(output), "--stages", "grouped_fp32", "--seed", "17"]
    )
    saved = sweep.SweepReport.model_validate_json(output.read_text())
    assert saved.source_snapshot is not None
    source = saved.source_snapshot.files["model.py"]
    assert source.content.encode("utf-8") == original
    assert source.sha256 == hashlib.sha256(original).hexdigest()
    assert saved.metadata.source_sha256 == {
        name: item.sha256 for name, item in saved.source_snapshot.files.items()
    }
    assert saved.source_snapshot.captured_at_utc.endswith("+00:00")
    assert "collect_profiles" not in json.loads(output.read_text())["metadata"]
    modal_calls.remote.assert_called_once_with(
        stages=["grouped_fp32"],
        rounds=3,
        trials=3,
        steps=20,
        seed=17,
        points=sweep.DEFAULT_POINTS,
        result_name="report.json",
    )


def test_experiment_failed_preflight_stops_before_weights(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = Mock(
        return_value=subprocess.CompletedProcess(
            args=["pytest"], returncode=1, stdout="correctness failed", stderr=""
        )
    )
    download = Mock(side_effect=AssertionError("must not download weights"))
    monkeypatch.setattr(sweep.subprocess, "run", preflight)
    monkeypatch.setattr(sweep, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="correctness failed"):
        sweep.experiment.local(
            stages=list(sweep.STAGES),
            rounds=1,
            trials=1,
            steps=1,
            seed=42,
            points=[(4, 1)],
        )
    preflight.assert_called_once()
    assert preflight.call_args.args[0] == [
        sys.executable,
        "-m",
        "pytest",
        *(f"/tests/{name}" for name in sweep.PREFLIGHT_TESTS),
        "-q",
    ]
    download.assert_not_called()


@pytest.mark.parametrize("heads", [12, 3, 1])
@pytest.mark.parametrize("backend", ["expanded", "repeat_kv", "grouped"])
def test_validate_model_prefill_decode_and_chunks(*, heads: int, backend: str) -> None:
    config = ModelConfig(
        vocab_size=23,
        d_model=24,
        max_seq_len=12,
        n_layers=1,
        num_heads=12,
        num_kv_heads=heads,
        attention_backend=backend,
    )
    model = TinyTransformer(config=config).eval()
    assert sweep.validate_model(model=model, config=config) <= 0.0001


@pytest.mark.parametrize("heads", [12, 3, 1])
def test_measure_forced_tokens_fresh_caches_sync_and_prefill_bytes(
    *, heads: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 12345)
    recorded_tokens = []
    for backend in ("expanded", "grouped"):
        events.clear()
        tokens: list[torch.Tensor] = []
        caches: list[KVCacheProtocol] = []
        lengths: list[int] = []
        config = ModelConfig(
            vocab_size=23,
            d_model=24,
            max_seq_len=12,
            n_layers=1,
            num_heads=12,
            num_kv_heads=heads,
            attention_backend=backend,
        )
        model = TinyTransformer(config=config).eval()

        def record_forward(
            module, args, kwargs, *, tokens=tokens, lengths=lengths, caches=caches
        ) -> None:
            token = args[0]
            cache = kwargs["cache"]
            events.append(f"forward:{token.shape[1]}")
            tokens.append(token.clone())
            lengths.append(cache.length)
            if token.shape[1] == 4:
                caches.append(cache)

        model.register_forward_pre_hook(record_forward, with_kwargs=True)
        result = sweep.measure(
            model=model,
            config=config,
            context=4,
            batch=2,
            trials=2,
            steps=2,
            seed=19,
        )
        assert len(result.latencies_ms) == 4
        assert len(result.setup_seconds) == 2
        assert lengths == [0, 4, 5, 6] * 2
        assert len(caches) == 2 and caches[0] is not caches[1]
        assert all(cache.length == 7 for cache in caches)
        bytes_per_position = 2 * config.n_layers * 2 * heads * 2 * 4
        assert result.cache_bytes_used == bytes_per_position * 4
        assert result.cache_bytes_allocated == bytes_per_position * 12
        assert result.peak_tensor_bytes == 12345
        assert result.tokens_per_second == pytest.approx(2000 / result.p50_ms)
        assert (
            events
            == [
                "sync",
                "forward:4",
                "forward:1",
                "sync",
                "sync",
                "forward:1",
                "sync",
                "sync",
                "forward:1",
                "sync",
            ]
            * 2
        )
        assert not torch.equal(tokens[0], tokens[4])
        recorded_tokens.append(tokens)
    for expanded, grouped in zip(*recorded_tokens, strict=True):
        torch.testing.assert_close(expanded, grouped, rtol=0, atol=0)


def test_main_saves_snapshot_beside_output_for_collect(
    *, tmp_path: Path, modal_calls: ModalCalls
) -> None:
    output = tmp_path / "report.json"
    sweep.main(argv=["--output", str(output)])
    saved = sweep.SweepReport.model_validate_json(output.read_text())
    snapshot = sweep.SourceSnapshot.model_validate_json(
        sweep.snapshot_path(output=output).read_text()
    )
    assert saved.source_snapshot == snapshot


def test_main_collect_reads_volume_without_launching(
    *,
    tmp_path: Path,
    modal_calls: ModalCalls,
    report: sweep.SweepReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    snapshot = sweep.capture_sources()
    sweep.snapshot_path(output=output).write_text(snapshot.model_dump_json())
    read_file = Mock(return_value=iter([report.model_dump_json().encode()]))
    monkeypatch.setattr(sweep.results_volume, "read_file", read_file)
    sweep.main(argv=["--output", str(output), "--collect"])
    read_file.assert_called_once_with("report.json")
    modal_calls.run.assert_not_called()
    modal_calls.remote.assert_not_called()
    saved = sweep.SweepReport.model_validate_json(output.read_text())
    assert saved.source_snapshot == snapshot
    assert saved.metadata.source_sha256 == {
        name: item.sha256 for name, item in snapshot.files.items()
    }
