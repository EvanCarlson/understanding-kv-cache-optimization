"""Benchmark FP32 cache variants and merge measured results into a JSON report."""

import argparse
import json
import logging
import os
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import modal
import numpy as np
import tiktoken
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from generate import (
    ATTENTION_BACKEND_CHOICES,
    GPT2_SMALL_D_MODEL,
    GPT2_SMALL_MAX_POSITIONS,
    GPT2_SMALL_N_LAYERS,
    GPT2_SMALL_NUM_HEADS,
    load_pretrained_model,
    validate_optimization_options,
)
from model import KVCacheFactory, KVCacheProtocol, ModelConfig, TinyTransformer
from shared import DEFAULT_SEED, Device, get_device, seed_everything

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LENGTHS = [64, 128, 256]
DEFAULT_BATCH_SIZES = [1, 8, 16, 64, 128]
DEFAULT_NUM_TRIALS = 10
DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_MODEL_REPO = "gpt2"
DEFAULT_CACHE_TYPE = "basic"
DEFAULT_GPU_TYPE = "A100"
CACHE_TYPE_CHOICES = ["basic", "gqa", "quantized", "paged", "none"]

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "benchmark_results.json"
DEFAULT_TRACES_DIR = RESULTS_DIR / "traces"


class LatencyPercentiles(BaseModel):
    model_config = ConfigDict(frozen=True)
    p50_ms: float
    p95_ms: float
    p99_ms: float


class BenchmarkPoint(BaseModel):
    model_config = ConfigDict(frozen=True)
    context_length: int
    batch_size: int
    # None when the point OOM'd before completing any trial.
    decode_latency: LatencyPercentiles | None
    throughput_tokens_per_sec: float | None
    num_trials: int
    # Both cache byte counts describe the prompt immediately after prefill.
    cache_bytes_used: int
    peak_memory_bytes: int
    cache_bytes_allocated: int = 0
    timed_steps: int = 0
    oom: bool
    # One list of step latencies per completed trial, kept so confidence intervals
    # can resample trials. Reports written before this field existed load as [].
    trial_latencies_ms: list[list[float]] = Field(default_factory=list)


class CacheBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True)
    cache_type: str
    gpu_type: str
    points: list[BenchmarkPoint]
    num_kv_heads: int = GPT2_SMALL_NUM_HEADS
    model_repo: str = DEFAULT_MODEL_REPO
    seed: int = DEFAULT_SEED
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_seq_len: int = 0
    torch_version: str = "unknown"
    dtype: str = "float32"
    attention_backend: str = "expanded"
    uptraining: bool = False


@dataclass(frozen=True)
class PrefillResult:
    """Decode-ready state with cache byte counts snapshotted before warmup."""

    cache: KVCacheProtocol | None
    running_ids: torch.Tensor
    next_token: torch.Tensor
    cache_bytes_used: int
    cache_bytes_allocated: int


def _sync_device(device: torch.device) -> None:
    if device.type == Device.CUDA:
        torch.cuda.synchronize()

    elif device.type == Device.MPS:
        torch.mps.synchronize()


def _release_cached_memory(device: torch.device) -> None:
    # The caching allocator does not return freed memory to
    # the driver on its own.
    if device.type == Device.CUDA:
        torch.cuda.empty_cache()

    elif device.type == Device.MPS:
        torch.mps.empty_cache()


def _load_results(path: Path) -> dict[str, CacheBenchmark]:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text())

    return {
        cache_type: CacheBenchmark.model_validate(entry)
        for cache_type, entry in raw.items()
    }


def _check_result_conflicts(
    *,
    path: Path,
    existing_keys: Collection[str],
    result_keys: Collection[str],
    overwrite: bool,
) -> None:
    conflicts = sorted(set(existing_keys).intersection(result_keys))
    if conflicts and not overwrite:
        raise FileExistsError(
            f"Result keys {', '.join(conflicts)} already exist in {path}; "
            "pass --overwrite to replace them or choose a different --output path"
        )


def _save_results(
    path: Path, results: dict[str, CacheBenchmark], *, overwrite: bool = False
) -> None:
    # Reload at write time: a long-running benchmark must not discard entries
    # added since preflight, or rewrite historical entries with schema defaults.
    payload = json.loads(path.read_text()) if path.exists() else {}
    _check_result_conflicts(
        path=path,
        existing_keys=payload.keys(),
        result_keys=results.keys(),
        overwrite=overwrite,
    )
    payload.update({key: benchmark.model_dump() for key, benchmark in results.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _result_key(benchmark: CacheBenchmark) -> str:
    key = (
        f"gqa_h{benchmark.num_kv_heads}"
        if benchmark.cache_type == "gqa"
        else benchmark.cache_type
    )
    if benchmark.attention_backend != "expanded" or benchmark.dtype != "float32":
        key += f"_{benchmark.attention_backend}_{benchmark.dtype}"
    return key


def _latency_percentiles(latencies_ms: list[float]) -> LatencyPercentiles:
    p50, p95, p99 = np.percentile(latencies_ms, [50, 95, 99])

    return LatencyPercentiles(p50_ms=float(p50), p95_ms=float(p95), p99_ms=float(p99))


def _create_cache(
    *, cache_type: str, config: ModelConfig, batch_size: int, device: torch.device
) -> KVCacheProtocol | None:
    if cache_type == "none":
        return None

    return KVCacheFactory.create(
        cache_type, config=config, batch_size=batch_size, device=device
    )


def _decode_step(
    *,
    model: TinyTransformer,
    cache: KVCacheProtocol | None,
    running_ids: torch.Tensor,
    next_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    running_ids = (
        torch.cat([running_ids, next_token], dim=1) if cache is None else next_token
    )
    logits = model(running_ids, cache=cache, logits_to_keep=1)

    return running_ids, logits


def _argmax_next_token(logits: torch.Tensor) -> torch.Tensor:
    result: torch.Tensor = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return result


def _prefill_and_warmup(
    *,
    model: TinyTransformer,
    cache_type: str,
    config: ModelConfig,
    context_length: int,
    batch_size: int,
    device: torch.device,
) -> PrefillResult:
    _release_cached_memory(device)
    cache = _create_cache(
        cache_type=cache_type, config=config, batch_size=batch_size, device=device
    )
    running_ids = torch.randint(
        0, config.vocab_size, (batch_size, context_length), device=device
    )

    with torch.no_grad():
        logits = model(running_ids, cache=cache, logits_to_keep=1)

    cache_bytes_used = cache.used_nbytes if cache is not None else 0
    cache_bytes_allocated = cache.allocated_nbytes if cache is not None else 0
    next_token = _argmax_next_token(logits)

    # Warmup: prime the decode-shaped (seq_len=1) kernels before timing.
    with torch.no_grad():
        running_ids, logits = _decode_step(
            model=model, cache=cache, running_ids=running_ids, next_token=next_token
        )

    next_token = _argmax_next_token(logits)

    return PrefillResult(
        cache=cache,
        running_ids=running_ids,
        next_token=next_token,
        cache_bytes_used=cache_bytes_used,
        cache_bytes_allocated=cache_bytes_allocated,
    )


def _all_eos(next_token: torch.Tensor, *, eos_token_id: int) -> bool:
    return bool((next_token == eos_token_id).all())


def _run_trial(
    *,
    model: TinyTransformer,
    cache_type: str,
    config: ModelConfig,
    context_length: int,
    batch_size: int,
    max_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
) -> tuple[list[float], int, int]:
    prefill = _prefill_and_warmup(
        model=model,
        cache_type=cache_type,
        config=config,
        context_length=context_length,
        batch_size=batch_size,
        device=device,
    )

    cache = prefill.cache
    running_ids, next_token = prefill.running_ids, prefill.next_token
    cache_bytes_used = prefill.cache_bytes_used
    cache_bytes_allocated = prefill.cache_bytes_allocated
    latencies_ms: list[float] = []

    if _all_eos(next_token, eos_token_id=eos_token_id):
        return latencies_ms, cache_bytes_used, cache_bytes_allocated

    for _ in range(max_new_tokens):
        _sync_device(device)
        start = time.perf_counter()

        with torch.no_grad():
            running_ids, logits = _decode_step(
                model=model,
                cache=cache,
                running_ids=running_ids,
                next_token=next_token,
            )

        _sync_device(device)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        next_token = _argmax_next_token(logits)

        if _all_eos(next_token, eos_token_id=eos_token_id):
            break

    return latencies_ms, cache_bytes_used, cache_bytes_allocated


def _run_sweep_point(
    *,
    model: TinyTransformer,
    cache_type: str,
    config: ModelConfig,
    context_length: int,
    batch_size: int,
    num_trials: int,
    max_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
) -> BenchmarkPoint:
    if device.type == Device.CUDA:
        torch.cuda.reset_peak_memory_stats(device)

    trial_latencies_ms: list[list[float]] = []
    cache_bytes_used = 0
    cache_bytes_allocated = 0
    completed_trials = 0
    oom = False

    for _ in range(num_trials):
        try:
            latencies_ms, cache_bytes_used, cache_bytes_allocated = _run_trial(
                model=model,
                cache_type=cache_type,
                config=config,
                context_length=context_length,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            # Report the OOM as a data point instead of crashing the sweep:
            # cache_type="none" recomputes full attention over the growing
            # sequence every step, so it's expected to run out of headroom
            # at large batch sizes well before the cached variants do.
            oom = True
            _release_cached_memory(device)
            break

        trial_latencies_ms.append(latencies_ms)
        completed_trials += 1

    peak_memory_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == Device.CUDA else 0
    )

    pooled_latencies_ms = [ms for trial in trial_latencies_ms for ms in trial]
    decode_latency = (
        _latency_percentiles(pooled_latencies_ms) if pooled_latencies_ms else None
    )
    # Tokens generated per second across the whole batch during
    # steady-state decode, using the per-step p50 as representative.
    throughput_tokens_per_sec = (
        batch_size / (decode_latency.p50_ms / 1000.0) if decode_latency else None
    )

    return BenchmarkPoint(
        context_length=context_length,
        batch_size=batch_size,
        decode_latency=decode_latency,
        throughput_tokens_per_sec=throughput_tokens_per_sec,
        num_trials=completed_trials,
        cache_bytes_used=cache_bytes_used,
        peak_memory_bytes=peak_memory_bytes,
        cache_bytes_allocated=cache_bytes_allocated,
        timed_steps=len(pooled_latencies_ms),
        oom=oom,
        trial_latencies_ms=trial_latencies_ms,
    )


# Profiler instrumentation adds per-op overhead so separate it from
# benchmarking the model.
def _capture_trace(
    *,
    model: TinyTransformer,
    cache_type: str,
    config: ModelConfig,
    context_length: int,
    device: torch.device,
) -> bytes:
    prefill = _prefill_and_warmup(
        model=model,
        cache_type=cache_type,
        config=config,
        context_length=context_length,
        batch_size=1,
        device=device,
    )

    activities = [ProfilerActivity.CPU]
    if device.type == Device.CUDA:
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=False) as prof, torch.no_grad():
        _decode_step(
            model=model,
            cache=prefill.cache,
            running_ids=prefill.running_ids,
            next_token=prefill.next_token,
        )

    trace_path = Path(f"/tmp/{cache_type}_trace.json")
    prof.export_chrome_trace(str(trace_path))

    return trace_path.read_bytes()


def run_benchmark(
    *,
    cache_type: str,
    context_lengths: list[int],
    batch_sizes: list[int],
    num_trials: int,
    max_new_tokens: int,
    model_repo: str,
    gpu_type: str,
    device: torch.device,
    capture_trace: bool,
    seed: int,
    num_kv_heads: int = GPT2_SMALL_NUM_HEADS,
    attention_backend: str = "expanded",
) -> tuple[CacheBenchmark, bytes | None]:
    validate_optimization_options(
        cache_type=cache_type, attention_backend=attention_backend
    )
    if num_kv_heads <= 0 or GPT2_SMALL_NUM_HEADS % num_kv_heads:
        raise ValueError("num_kv_heads must be a positive divisor of 12")
    if cache_type != "gqa" and num_kv_heads != GPT2_SMALL_NUM_HEADS:
        raise ValueError("Reduced KV heads require cache_type='gqa'")
    seed_everything(seed=seed)
    tokenizer = tiktoken.get_encoding("gpt2")

    max_context_length = max(context_lengths)
    # Each trial: 1 warmup step + up to max_new_tokens timed steps beyond the
    # longest prefill.
    #
    # Trace capture: a separate 1 warmup + 1 traced step beyond the longest
    # prefill.
    trial_positions = max_new_tokens + 1
    trace_positions = 2 if capture_trace else 0
    needed_positions = max(trial_positions, trace_positions)
    max_seq_len = max_context_length + needed_positions

    if max_seq_len > GPT2_SMALL_MAX_POSITIONS:
        raise ValueError(
            f"max_context_length ({max_context_length}) + needed_positions "
            f"({needed_positions}) = {max_seq_len} exceeds gpt2's positional "
            f"embedding limit of {GPT2_SMALL_MAX_POSITIONS}. needed_positions "
            f"is max(max_new_tokens + 1 ({trial_positions}), "
            f"trace_positions ({trace_positions}))"
        )

    config = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        d_model=GPT2_SMALL_D_MODEL,
        n_layers=GPT2_SMALL_N_LAYERS,
        num_heads=GPT2_SMALL_NUM_HEADS,
        num_kv_heads=num_kv_heads,
        attention_backend=attention_backend,
        max_seq_len=max_seq_len,
    )
    model = load_pretrained_model(config=config, model_repo=model_repo, device=device)

    points: list[BenchmarkPoint] = []
    eos_token_id = tokenizer.eot_token
    sweep_points = [
        (context_length, batch_size)
        for context_length in context_lengths
        for batch_size in batch_sizes
    ]

    for context_length, batch_size in tqdm(sweep_points, desc=f"{cache_type} sweep"):
        seed_everything(seed=seed)
        point = _run_sweep_point(
            model=model,
            cache_type=cache_type,
            config=config,
            context_length=context_length,
            batch_size=batch_size,
            num_trials=num_trials,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
        )
        points.append(point)

        if point.oom:
            logger.info(
                "cache_type=%s context_length=%d batch_size=%d OOM after "
                "%d/%d trials, peak_memory_bytes=%d",
                cache_type,
                context_length,
                batch_size,
                point.num_trials,
                num_trials,
                point.peak_memory_bytes,
            )
        else:
            assert point.decode_latency is not None
            assert point.throughput_tokens_per_sec is not None
            logger.info(
                "cache_type=%s context_length=%d batch_size=%d num_trials=%d "
                "latency_p50_ms=%.3f latency_p95_ms=%.3f latency_p99_ms=%.3f "
                "throughput_tokens_per_sec=%.1f cache_bytes_used=%d "
                "peak_memory_bytes=%d",
                cache_type,
                context_length,
                batch_size,
                point.num_trials,
                point.decode_latency.p50_ms,
                point.decode_latency.p95_ms,
                point.decode_latency.p99_ms,
                point.throughput_tokens_per_sec,
                point.cache_bytes_used,
                point.peak_memory_bytes,
            )

    trace_bytes = (
        _capture_trace(
            model=model,
            cache_type=cache_type,
            config=config,
            context_length=max_context_length,
            device=device,
        )
        if capture_trace
        else None
    )

    return CacheBenchmark(
        cache_type=cache_type,
        gpu_type=gpu_type,
        points=points,
        num_kv_heads=num_kv_heads,
        model_repo=model_repo,
        seed=seed,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        torch_version=torch.__version__,
        attention_backend=attention_backend,
    ), trace_bytes


app = modal.App("understanding-kv-cache-optimization-benchmark")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .workdir("/")
    .add_local_python_source("generate", "model", "shared")
)


hf_token = os.environ.get("HF_TOKEN")
hf_secrets = [modal.Secret.from_dict({"HF_TOKEN": hf_token})] if hf_token else []


@app.function(image=image, gpu=DEFAULT_GPU_TYPE, timeout=1800, secrets=hf_secrets)
def remote_run_benchmark(
    *,
    cache_type: str,
    context_lengths: list[int],
    batch_sizes: list[int],
    num_trials: int,
    max_new_tokens: int,
    model_repo: str,
    gpu_type: str,
    capture_trace: bool,
    seed: int,
    num_kv_heads: int = GPT2_SMALL_NUM_HEADS,
    attention_backend: str = "expanded",
) -> tuple[dict, bytes | None]:
    benchmark, trace_bytes = run_benchmark(
        cache_type=cache_type,
        context_lengths=context_lengths,
        batch_sizes=batch_sizes,
        num_trials=num_trials,
        max_new_tokens=max_new_tokens,
        model_repo=model_repo,
        gpu_type=gpu_type,
        device=torch.device("cuda"),
        capture_trace=capture_trace,
        seed=seed,
        num_kv_heads=num_kv_heads,
        attention_backend=attention_backend,
    )

    return benchmark.model_dump(), trace_bytes


def _parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-type",
        default=DEFAULT_CACHE_TYPE,
        choices=CACHE_TYPE_CHOICES,
        help="Which cache to benchmark",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        choices=[1, 2, 3, 4, 6, 12],
        help="KV heads per layer (default: 3 for gqa, 12 otherwise)",
    )
    parser.add_argument(
        "--context-lengths",
        type=_parse_int_list,
        default=DEFAULT_CONTEXT_LENGTHS,
        help="Comma-separated context lengths to sweep, e.g. 64,128,256,512",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_int_list,
        default=DEFAULT_BATCH_SIZES,
        help=(
            "Comma-separated batch sizes to sweep, e.g. 1,8,32. Larger "
            "batches amortize per-step kernel-launch/dispatch overhead, so "
            "cache-vs-no-cache and cache-type comparisons can look "
            "different at batch_size=1 than at production-like batch sizes"
        ),
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=(
            "Independent prefill+decode trials run at each context "
            "length/batch size, each with its own fresh cache"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Max decode steps per trial, after 1 untimed warmup step (a trial "
        "stops earlier if the model predicts the EOS token)",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repo for GPT-2-shaped weights",
    )
    parser.add_argument(
        "--modal", action="store_true", help="Run on a Modal-provisioned remote GPU"
    )
    parser.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help="Modal GPU type to use when --modal is set",
    )
    parser.add_argument(
        "--no-trace", action="store_true", help="Skip capturing a torch.profiler trace"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Shared results JSON path (existing keys require --overwrite)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing this benchmark's existing result key in --output",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=DEFAULT_TRACES_DIR,
        help="Directory for trace JSON files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for reproducible benchmarking",
    )

    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="expanded",
        help="Attention implementation (expanded or grouped, both FP32)",
    )
    args = parser.parse_args(argv)
    try:
        validate_optimization_options(
            cache_type=args.cache_type,
            attention_backend=args.attention_backend,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.num_kv_heads is None:
        args.num_kv_heads = 3 if args.cache_type == "gqa" else GPT2_SMALL_NUM_HEADS
    if args.cache_type != "gqa" and args.num_kv_heads != GPT2_SMALL_NUM_HEADS:
        parser.error("Reduced KV heads require --cache-type gqa")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    requested = CacheBenchmark(
        cache_type=args.cache_type,
        num_kv_heads=args.num_kv_heads,
        attention_backend=args.attention_backend,
        gpu_type=args.gpu_type,
        points=[],
    )
    _check_result_conflicts(
        path=args.output,
        existing_keys=_load_results(args.output).keys(),
        result_keys={_result_key(requested)},
        overwrite=args.overwrite,
    )
    capture_trace = not args.no_trace

    if args.modal:
        with modal.enable_output(), app.run():
            benchmark_dict, trace_bytes = remote_run_benchmark.with_options(
                gpu=args.gpu_type
            ).remote(
                cache_type=args.cache_type,
                context_lengths=args.context_lengths,
                batch_sizes=args.batch_sizes,
                num_trials=args.num_trials,
                max_new_tokens=args.max_new_tokens,
                model_repo=args.model_repo,
                gpu_type=args.gpu_type,
                capture_trace=capture_trace,
                seed=args.seed,
                num_kv_heads=args.num_kv_heads,
                attention_backend=args.attention_backend,
            )

        benchmark = CacheBenchmark.model_validate(benchmark_dict)
    else:
        device = torch.device(get_device())
        benchmark, trace_bytes = run_benchmark(
            cache_type=args.cache_type,
            context_lengths=args.context_lengths,
            batch_sizes=args.batch_sizes,
            num_trials=args.num_trials,
            max_new_tokens=args.max_new_tokens,
            model_repo=args.model_repo,
            gpu_type=str(device),
            device=device,
            capture_trace=capture_trace,
            seed=args.seed,
            num_kv_heads=args.num_kv_heads,
            attention_backend=args.attention_backend,
        )

    result_key = _result_key(benchmark)
    _save_results(args.output, {result_key: benchmark}, overwrite=args.overwrite)
    logger.info("Wrote %s entry to %s", result_key, args.output)

    if trace_bytes is not None:
        args.traces_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.traces_dir / f"{result_key}.json"
        trace_path.write_bytes(trace_bytes)
        logger.info("Wrote trace to %s", trace_path)

    print(f"\n=== {args.cache_type} (GPU: {benchmark.gpu_type}) ===")
    print(
        f"{'context_length':>14}  {'batch_size':>10}  {'num_trials':>10}  "
        f"{'p50_ms':>10}  {'p95_ms':>10}  {'p99_ms':>10}  {'tokens/sec':>10}  "
        f"{'cache_bytes_used':>16}  {'peak_mem_mb':>12}"
    )

    for point in benchmark.points:
        peak_mem_mb = point.peak_memory_bytes / (1024 * 1024)

        if point.oom:
            print(
                f"{point.context_length:>14}  {point.batch_size:>10}  "
                f"{point.num_trials:>10}  {'OOM':>10}  {'OOM':>10}  {'OOM':>10}  "
                f"{'OOM':>10}  {point.cache_bytes_used:>16}  {peak_mem_mb:>12.1f}"
            )
            continue

        assert point.decode_latency is not None
        assert point.throughput_tokens_per_sec is not None
        print(
            f"{point.context_length:>14}  {point.batch_size:>10}  "
            f"{point.num_trials:>10}  "
            f"{point.decode_latency.p50_ms:>10.3f}  {point.decode_latency.p95_ms:>10.3f}  "
            f"{point.decode_latency.p99_ms:>10.3f}  "
            f"{point.throughput_tokens_per_sec:>10.1f}  {point.cache_bytes_used:>16}  "
            f"{peak_mem_mb:>12.1f}"
        )


if __name__ == "__main__":
    main()
