"""Compare expanded/grouped FP32 MHA, GQA, and MQA on a Modal A100.

Accept a context/batch sweep and write timings, failures, and a pre-launch source
snapshot to a new JSON report. Existing reports require explicit --overwrite.
"""

import argparse
import gc
import hashlib
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import modal
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file

from benchmark import hf_secrets
from generate import (
    GPT2_SMALL_D_MODEL,
    GPT2_SMALL_MAX_POSITIONS,
    GPT2_SMALL_N_LAYERS,
    GPT2_SMALL_NUM_HEADS,
)
from model import (
    KVCacheFactory,
    KVCacheProtocol,
    ModelConfig,
    TinyTransformer,
    load_gpt2_weights,
)
from shared import DEFAULT_SEED, seed_everything

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
STAGES = {
    "expanded_fp32": "expanded",
    "grouped_fp32": "grouped",
    "repeat_kv_fp32": "repeat_kv",
}
DEFAULT_STAGES = ["expanded_fp32", "grouped_fp32"]
DEFAULT_POINTS = [(1000, 1), (256, 8), (256, 32)]
DEFAULT_ROUNDS = 3
DEFAULT_TRIALS = 3
DEFAULT_STEPS = 20
PREFLIGHT_TESTS = (
    "test_attention_optimizations.py",
    "test_benchmark.py",
    "test_gqa_conversion.py",
    "test_kv_cache.py",
    "test_mha.py",
    "test_optimization_cli.py",
)
RESULTS_VOLUME_NAME = "gqa-optimization-results"
RESULTS_MOUNT = "/results"
app = modal.App("gqa-optimization-experiments")
# A detached run keeps going if the local client disconnects (a laptop sleeping
# drops Modal's heartbeat and stops a non-detached app), so the finished report is
# also written here and fetched later with --collect.
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
experiment_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(SCRIPT_DIR))
    .workdir("/")
    .add_local_python_source("benchmark", "generate", "model", "shared")
)
for test_name in ("conftest.py", *PREFLIGHT_TESTS):
    experiment_image = experiment_image.add_local_file(
        SCRIPT_DIR / "tests" / test_name, f"/tests/{test_name}"
    )


class SourceFile(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    sha256: str
    content: str


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    captured_at_utc: str
    files: dict[str, SourceFile]


class Measurement(BaseModel):
    model_config = ConfigDict(frozen=True)
    context_length: int
    batch_size: int
    latencies_ms: list[float]
    p50_ms: float
    p95_ms: float
    tokens_per_second: float
    setup_seconds: list[float]
    cache_bytes_used: int
    cache_bytes_allocated: int
    peak_tensor_bytes: int


class SweepRun(Measurement):
    round: int
    heads: int
    stage: str
    logit_relative_rmse: float


class SweepError(BaseModel):
    model_config = ConfigDict(frozen=True)
    round: int
    heads: int
    stage: str
    context_length: int | None
    batch_size: int | None
    oom: bool
    error: str


class SweepMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    gpu: str
    gpu_uuid: str
    torch: str
    cuda: str | None
    cpu_threads: int
    rounds: int
    trials: int
    steps: int
    seed: int
    stages: list[str]
    points: list[tuple[int, int]]
    forced_tokens: bool = True
    capacity: int = GPT2_SMALL_MAX_POSITIONS
    dtype: str = "float32"
    single_container: bool = True
    tests: str
    checkpoint_sha256: str
    source_sha256: dict[str, str] = Field(default_factory=dict)


class SweepReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    metadata: SweepMetadata
    runs: list[SweepRun] = Field(default_factory=list)
    errors: list[SweepError] = Field(default_factory=list)
    source_snapshot: SourceSnapshot | None = None


def capture_sources() -> SourceSnapshot:
    """Resolve imported modules, not same-named files in the working directory."""
    paths = {}
    for name in (__name__, "benchmark", "generate", "model", "shared"):
        path = Path(inspect.getfile(sys.modules[name])).resolve()
        paths[path.name] = path
    paths.update({name: SCRIPT_DIR / name for name in ("pyproject.toml", "uv.lock")})
    paths.update(
        {
            f"tests/{name}": SCRIPT_DIR / "tests" / name
            for name in ("conftest.py", *PREFLIGHT_TESTS)
        }
    )
    files = {}
    for name, path in paths.items():
        content = path.read_bytes()
        files[name] = SourceFile(
            path=str(path),
            sha256=hashlib.sha256(content).hexdigest(),
            content=content.decode("utf-8"),
        )
    return SourceSnapshot(captured_at_utc=datetime.now(UTC).isoformat(), files=files)


def make_cache(
    *, config: ModelConfig, batch: int, device: torch.device
) -> KVCacheProtocol:
    return KVCacheFactory.create("gqa", config=config, batch_size=batch, device=device)


@torch.no_grad()
def validate_model(*, model: TinyTransformer, config: ModelConfig) -> float:
    device = next(model.parameters()).device
    tokens = torch.arange(12, device=device).reshape(1, 12)
    reference_config = replace(config, attention_backend="expanded")
    reference = (
        TinyTransformer(config=reference_config)
        .to(device=device, dtype=torch.float32)
        .eval()
    )
    reference.load_state_dict(model.state_dict())
    expected_cache = make_cache(config=reference_config, batch=1, device=device)
    actual_cache = make_cache(config=config, batch=1, device=device)
    errors = []
    for start, end in [(0, 6), (6, 7), (7, 10), (10, 11), (11, 12)]:
        expected = reference(tokens[:, start:end], cache=expected_cache)
        actual = model(tokens[:, start:end], cache=actual_cache)
        relative_rmse = (
            (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()
        ).item()
        if not math.isfinite(relative_rmse) or relative_rmse > 0.0001:
            raise AssertionError(
                f"Logit relative RMSE {relative_rmse} exceeds tolerance"
            )
        errors.append(relative_rmse)
    return max(errors)


@torch.no_grad()
def measure(
    *,
    model: TinyTransformer,
    config: ModelConfig,
    context: int,
    batch: int,
    trials: int,
    steps: int,
    seed: int,
) -> Measurement:
    device = next(model.parameters()).device
    elapsed = []
    setup_seconds = []
    cache_used = cache_allocated = 0
    torch.cuda.reset_peak_memory_stats()
    for trial in range(trials):
        seed_everything(seed=seed + trial)
        prompt = torch.randint(0, config.vocab_size, (batch, context), device=device)
        continuation = torch.randint(
            0, config.vocab_size, (steps + 1, batch, 1), device=device
        )
        torch.cuda.synchronize()
        setup_start = time.perf_counter()
        cache = make_cache(config=config, batch=batch, device=device)
        model(prompt, cache=cache, logits_to_keep=1)
        # Used bytes describe the prompt, before warmup or timed decode grows it.
        cache_used = cache.used_nbytes
        cache_allocated = cache.allocated_nbytes
        model(continuation[0], cache=cache)
        torch.cuda.synchronize()
        setup_seconds.append(time.perf_counter() - setup_start)

        for token in continuation[1:]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(token, cache=cache)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
        del cache
    p50 = float(np.percentile(elapsed, 50))
    return Measurement(
        context_length=context,
        batch_size=batch,
        latencies_ms=elapsed,
        p50_ms=p50,
        p95_ms=float(np.percentile(elapsed, 95)),
        tokens_per_second=batch * 1000 / p50,
        setup_seconds=setup_seconds,
        cache_bytes_used=cache_used,
        cache_bytes_allocated=cache_allocated,
        peak_tensor_bytes=torch.cuda.max_memory_allocated(),
    )


@app.function(
    image=experiment_image,
    gpu="A100",
    timeout=3600,
    secrets=hf_secrets,
    volumes={RESULTS_MOUNT: results_volume},
)
def experiment(
    *,
    stages: list[str],
    rounds: int,
    trials: int,
    steps: int,
    seed: int,
    points: list[tuple[int, int]],
    result_name: str | None = None,
) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *(f"/tests/{name}" for name in PREFLIGHT_TESTS),
            "-q",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": "/root:/"},
    )
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)
    weights_path = hf_hub_download(repo_id="gpt2", filename="model.safetensors")
    weights = load_file(weights_path, device="cpu")
    props = torch.cuda.get_device_properties(0)
    report = SweepReport(
        metadata=SweepMetadata(
            gpu=props.name,
            gpu_uuid=str(props.uuid),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cpu_threads=torch.get_num_threads(),
            rounds=rounds,
            trials=trials,
            steps=steps,
            seed=seed,
            stages=stages,
            points=points,
            tests=tests.stdout,
            checkpoint_sha256=hashlib.sha256(
                Path(weights_path).read_bytes()
            ).hexdigest(),
        )
    )
    for round_idx in range(rounds):
        order = stages if round_idx % 2 == 0 else list(reversed(stages))
        for heads in [12, 3, 1] if round_idx % 2 == 0 else [1, 3, 12]:
            for stage in order:
                logger.info("round=%s heads=%s stage=%s", round_idx + 1, heads, stage)
                config = ModelConfig(
                    vocab_size=50257,
                    d_model=GPT2_SMALL_D_MODEL,
                    max_seq_len=GPT2_SMALL_MAX_POSITIONS,
                    n_layers=GPT2_SMALL_N_LAYERS,
                    num_heads=GPT2_SMALL_NUM_HEADS,
                    num_kv_heads=heads,
                    attention_backend=STAGES[stage],
                )
                gc.collect()
                torch.cuda.empty_cache()
                model = None
                context = batch = None
                try:
                    seed_everything(seed=seed)
                    model = TinyTransformer(config=config).eval()
                    load_gpt2_weights(model, gpt2_state_dict=weights)
                    model = model.to("cuda", dtype=torch.float32)
                    error = validate_model(model=model, config=config)
                    for context, batch in points:
                        try:
                            point = measure(
                                model=model,
                                config=config,
                                context=context,
                                batch=batch,
                                trials=trials,
                                steps=steps,
                                seed=seed,
                            )
                        except torch.cuda.OutOfMemoryError as exc:
                            failure = SweepError(
                                round=round_idx + 1,
                                heads=heads,
                                stage=stage,
                                context_length=context,
                                batch_size=batch,
                                oom=True,
                                error=str(exc),
                            )
                            report.errors.append(failure)
                            logger.warning("%s", failure.model_dump_json())
                            point = None
                        if point is None:
                            gc.collect()
                            torch.cuda.empty_cache()
                            continue
                        row = SweepRun(
                            round=round_idx + 1,
                            heads=heads,
                            stage=stage,
                            logit_relative_rmse=error,
                            **point.model_dump(),
                        )
                        report.runs.append(row)
                        logger.info("%s", row.model_dump_json(exclude={"latencies_ms"}))
                except (RuntimeError, AssertionError, ValueError) as exc:
                    failure = SweepError(
                        round=round_idx + 1,
                        heads=heads,
                        stage=stage,
                        context_length=context,
                        batch_size=batch,
                        oom=isinstance(exc, torch.cuda.OutOfMemoryError),
                        error=repr(exc),
                    )
                    report.errors.append(failure)
                    logger.warning("%s", failure.model_dump_json())
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
    payload = report.model_dump(mode="json")
    if result_name is not None:
        Path(RESULTS_MOUNT, result_name).write_text(json.dumps(payload))
        results_volume.commit()

    return payload


def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=DEFAULT_STAGES,
        help=(
            "FP32 attention implementations to compare (default: expanded_fp32 "
            "and grouped_fp32; repeat_kv_fp32 is opt-in)"
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Rounds, reversing order each round",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Fresh-cache trials per point",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Timed decode steps after one warmup step",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Model seed; trial tokens use seed + trial index",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON report, including source snapshot",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing report",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help=(
            "Fetch a finished run's report from the Modal volume instead of "
            "launching one; needs the .snapshot.json the launch saved"
        ),
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        help="Prompt lengths; requires --batch-sizes",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes crossed with all --context-lengths",
    )
    args = parser.parse_args(argv)
    if min(args.rounds, args.trials, args.steps) < 1:
        parser.error("Rounds, trials, and steps must be positive")
    if (args.context_lengths is None) != (args.batch_sizes is None):
        parser.error("--context-lengths and --batch-sizes must be specified together")
    args.points = DEFAULT_POINTS
    if args.context_lengths is not None:
        if any(value < 1 for value in args.context_lengths + args.batch_sizes):
            parser.error("Context lengths and batch sizes must be positive")
        args.points = [
            (context, batch)
            for context in args.context_lengths
            for batch in args.batch_sizes
        ]
    if (
        max(context for context, _ in args.points) + 1 + args.steps
        > GPT2_SMALL_MAX_POSITIONS
    ):
        parser.error("Prefill, warmup, and timed steps must fit 1024 positions")
    if not args.overwrite and (args.output.exists() or args.output.is_symlink()):
        parser.error(
            f"Output already exists: {args.output}; use --overwrite to replace it"
        )
    if args.output.is_dir():
        parser.error(f"Output is a directory: {args.output}")
    return args


def snapshot_path(*, output: Path) -> Path:
    return output.with_name(output.name + ".snapshot.json")


def read_volume_report(*, name: str) -> SweepReport:
    return SweepReport.model_validate_json(b"".join(results_volume.read_file(name)))


def finalize_report(
    *, report: SweepReport, snapshot: SourceSnapshot, output: Path, overwrite: bool
) -> None:
    metadata = report.metadata.model_copy(
        update={
            "source_sha256": {
                name: source.sha256 for name, source in snapshot.files.items()
            }
        }
    )
    report = report.model_copy(
        update={"metadata": metadata, "source_snapshot": snapshot}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
    print(f"Saved {output}; {len(report.errors)} failed configurations")


def main(*, argv: list[str] | None = None) -> None:
    args = parse_args(argv=argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    saved_snapshot = snapshot_path(output=args.output)
    if args.collect:
        snapshot = SourceSnapshot.model_validate_json(saved_snapshot.read_text())
        report = read_volume_report(name=args.output.name)
    else:
        snapshot = capture_sources()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        saved_snapshot.write_text(snapshot.model_dump_json())
        with modal.enable_output(), app.run(detach=True):
            report = SweepReport.model_validate(
                experiment.remote(
                    stages=args.stages,
                    rounds=args.rounds,
                    trials=args.trials,
                    steps=args.steps,
                    seed=args.seed,
                    points=args.points,
                    result_name=args.output.name,
                )
            )
    finalize_report(
        report=report, snapshot=snapshot, output=args.output, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
