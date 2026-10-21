"""Compare basic KV caching with full recomputation in one Modal A100 allocation.

Uses benchmark._run_sweep_point unchanged: generated tokens, EOS stopping, one
untimed warmup, and synchronized decode timing excluding argmax/EOS checks.
Reports include a pre-launch source snapshot and never overwrite existing paths.
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal
import tiktoken
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import benchmark
from generate import (
    GPT2_SMALL_D_MODEL,
    GPT2_SMALL_MAX_POSITIONS,
    GPT2_SMALL_N_LAYERS,
    GPT2_SMALL_NUM_HEADS,
)
from model import ModelConfig, TinyTransformer, load_gpt2_weights
from shared import DEFAULT_SEED, seed_everything

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POINTS = [
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
VARIANTS = ("basic", "none")
PREFLIGHT_TESTS = (
    "test_attention_optimizations.py",
    "test_benchmark.py",
    "test_gqa_conversion.py",
    "test_kv_cache.py",
    "test_mha.py",
    "test_optimization_cli.py",
)
app = modal.App("kv-cache-baseline")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(SCRIPT_DIR))
    .workdir("/")
    .add_local_python_source("benchmark", "generate", "model", "shared")
)
for test_name in ("conftest.py", *PREFLIGHT_TESTS):
    image = image.add_local_file(
        SCRIPT_DIR / "tests" / test_name, f"/tests/{test_name}"
    )


def capture_sources() -> dict:
    """Snapshot imported source paths, not lookalikes in the working directory."""
    paths = {}
    for name in (__name__, "benchmark", "generate", "model", "shared"):
        path = Path(inspect.getfile(sys.modules[name])).resolve()
        paths[path.name] = path
    paths.update({name: SCRIPT_DIR / name for name in ("pyproject.toml", "uv.lock")})
    paths.update(
        {
            f"tests/{name}": SCRIPT_DIR / "tests" / name
            for name in ("conftest.py", *PREFLIGHT_TESTS, "test_baseline_sweep.py")
        }
    )
    files = {}
    for name, path in paths.items():
        content = path.read_bytes()
        files[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "content": content.decode("utf-8"),
        }
    return {"captured_at_utc": datetime.now(UTC).isoformat(), "files": files}


def validate_sweep(*, points: list[tuple[int, int]], trials: int, steps: int) -> None:
    if trials < 1 or steps < 1:
        raise ValueError("Trials and steps must be positive")
    if not points or any(context < 1 or batch < 1 for context, batch in points):
        raise ValueError("Points must have positive context lengths and batch sizes")
    if max(context for context, _ in points) + 1 + steps > GPT2_SMALL_MAX_POSITIONS:
        raise ValueError("Prefill, one warmup, and timed steps must fit 1024 positions")


@app.function(image=image, gpu="A100", timeout=1800, secrets=benchmark.hf_secrets)
def experiment(
    *, points: list[tuple[int, int]], trials: int, steps: int, seed: int
) -> dict:
    validate_sweep(points=points, trials=trials, steps=steps)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    started_at = datetime.now(UTC).isoformat()
    torch.set_num_threads(1)
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
        env={
            **os.environ,
            "PYTHONPATH": "/root:/",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    if tests.returncode:
        raise RuntimeError(tests.stdout + tests.stderr)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    seed_everything(seed=seed)
    tokenizer = tiktoken.get_encoding("gpt2")
    config = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        d_model=GPT2_SMALL_D_MODEL,
        n_layers=GPT2_SMALL_N_LAYERS,
        num_heads=GPT2_SMALL_NUM_HEADS,
        num_kv_heads=GPT2_SMALL_NUM_HEADS,
        attention_backend="grouped",
        max_seq_len=GPT2_SMALL_MAX_POSITIONS,
    )
    weights_path = Path(
        hf_hub_download(
            repo_id=benchmark.DEFAULT_MODEL_REPO, filename="model.safetensors"
        )
    )
    with weights_path.open("rb") as weights_file:
        checkpoint_sha256 = hashlib.file_digest(weights_file, "sha256").hexdigest()
    # Load the exact file hashed above, without resolving a moving Hub ref twice.
    model = TinyTransformer(config=config).to(device=device, dtype=torch.float32)
    model.eval()
    load_gpt2_weights(
        model, gpt2_state_dict=load_file(str(weights_path), device=str(device))
    )

    results = {
        variant: benchmark.CacheBenchmark(
            cache_type=variant,
            gpu_type=props.name,
            points=[],
            num_kv_heads=config.num_kv_heads,
            model_repo=benchmark.DEFAULT_MODEL_REPO,
            seed=seed,
            max_new_tokens=steps,
            max_seq_len=config.max_seq_len,
            torch_version=torch.__version__,
            dtype="float32",
            attention_backend=config.attention_backend,
        )
        for variant in VARIANTS
    }
    execution_order = []
    for index, (context, batch) in enumerate(points):
        order = VARIANTS if index % 2 == 0 else VARIANTS[::-1]
        for variant in order:
            execution_order.append(
                {"context_length": context, "batch_size": batch, "cache_type": variant}
            )
            seed_everything(seed=seed)
            point = benchmark._run_sweep_point(
                model=model,
                cache_type=variant,
                config=config,
                context_length=context,
                batch_size=batch,
                num_trials=trials,
                max_new_tokens=steps,
                eos_token_id=tokenizer.eot_token,
                device=device,
            )
            results[variant].points.append(point)
            logger.info(
                "%s %s",
                variant,
                point.model_dump_json(exclude={"trial_latencies_ms"}),
            )

    return {
        **{
            variant: result.model_dump(mode="json")
            for variant, result in results.items()
        },
        "metadata": {
            "gpu": props.name,
            "gpu_uuid": str(props.uuid),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "seed": seed,
            "trials": trials,
            "steps": steps,
            "warmup_steps": 1,
            "forced_tokens": False,
            "eos_token_id": tokenizer.eot_token,
            "max_seq_len": config.max_seq_len,
            "dtype": "float32",
            "attention_backend": config.attention_backend,
            "model_repo": benchmark.DEFAULT_MODEL_REPO,
            "single_container": True,
            "points": points,
            "execution_order": execution_order,
            "tests": tests.stdout + tests.stderr,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }


def _parse_point(value: str) -> tuple[int, int]:
    try:
        context, batch = value.split(":")
        return int(context), int(batch)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected CONTEXT:BATCH, e.g. 256:8"
        ) from error


def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON report path (never overwritten)",
    )
    parser.add_argument(
        "--points",
        type=_parse_point,
        nargs="+",
        metavar="CONTEXT:BATCH",
        default=DEFAULT_POINTS,
        help="Exact ordered points, e.g. 64:1 1000:1 256:8 (not a Cartesian product)",
    )
    parser.add_argument("--trials", type=int, default=benchmark.DEFAULT_NUM_TRIALS)
    parser.add_argument(
        "--steps",
        type=int,
        default=benchmark.DEFAULT_MAX_NEW_TOKENS,
        help="Maximum timed decode steps after one warmup; EOS can stop a trial early",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        validate_sweep(points=args.points, trials=args.trials, steps=args.steps)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists() or args.output.is_symlink():
        parser.error(
            f"Output already exists: {args.output}; choose a new --output path"
        )
    return args


def main(*, argv: list[str] | None = None) -> None:
    args = parse_args(argv=argv)
    snapshot = capture_sources()
    with modal.enable_output(), app.run():
        report = experiment.remote(
            points=args.points, trials=args.trials, steps=args.steps, seed=args.seed
        )
    report["source_snapshot"] = snapshot
    report["metadata"]["source_sha256"] = {
        name: source["sha256"] for name, source in snapshot["files"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects reports created while the GPU job ran.
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
