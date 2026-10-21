"""Profile the CUDA kernels in one decode attention step that uses repeat_kv.

For 12, 3, and 1 KV heads, times the attention matmuls and softmax on an A100
after expanding K/V with Hugging Face's repeat_kv, and lists the CUDA kernels
that ran. For 1 KV head repeat_kv returns a stride-0 view without copying, and the
copy that matmul then needs shows up as copy kernels inside the matmuls instead.
Writes a JSON report; launches a billed Modal A100 job.
"""

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import modal
import torch
from pydantic import BaseModel, ConfigDict
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from transformers.models.llama.modeling_llama import repeat_kv

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH = 128
DEFAULT_SEQ = 533
DEFAULT_OUTPUT = "results/repeat-kv-attention-kernels-2026-10-06.json"
KV_HEADS = [12, 3, 1]
NUM_QUERY_HEADS = 12
HEAD_DIM = 64
CACHE_POSITIONS = 1024
WARMUP_STEPS = 5
TIMED_STEPS = 30
app = modal.App("repeat-kv-attention-kernels")
image = modal.Image.debian_slim(python_version="3.12").uv_sync(
    uv_project_dir=str(SCRIPT_DIR)
)


class KernelTime(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    count: int
    cuda_us: float


class KvHeadsProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    kv_heads: int
    step_ms_p50: float
    kernels: list[KernelTime]


class KernelReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    gpu: str
    torch: str
    batch_size: int
    seq_len: int
    profiles: list[KvHeadsProfile]


def _attention_step(
    *,
    cache_keys: torch.Tensor,
    cache_values: torch.Tensor,
    queries: torch.Tensor,
    seq: int,
    n_rep: int,
) -> torch.Tensor:
    keys = repeat_kv(cache_keys[:, :, :seq, :], n_rep)
    values = repeat_kv(cache_values[:, :, :seq, :], n_rep)
    scores = torch.matmul(queries, keys.transpose(2, 3)) / HEAD_DIM**0.5

    return torch.matmul(torch.softmax(scores, dim=-1), values)


@app.function(image=image, gpu="A100-40GB", timeout=1200)
def profile(*, batch: int, seq: int) -> dict:
    profiles = []
    for kv_heads in KV_HEADS:
        n_rep = NUM_QUERY_HEADS // kv_heads
        cache_keys = torch.randn(
            batch, kv_heads, CACHE_POSITIONS, HEAD_DIM, device="cuda"
        )
        cache_values = torch.randn_like(cache_keys)
        queries = torch.randn(batch, NUM_QUERY_HEADS, 1, HEAD_DIM, device="cuda")

        step = partial(
            _attention_step,
            cache_keys=cache_keys,
            cache_values=cache_values,
            queries=queries,
            seq=seq,
            n_rep=n_rep,
        )

        for _ in range(WARMUP_STEPS):
            step()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_STEPS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(TIMED_STEPS)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            step()
            end.record()
        torch.cuda.synchronize()
        step_ms = sorted(
            start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
        )

        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        kernels = sorted(
            (
                KernelTime(
                    name=event.key[:100],
                    count=event.count,
                    cuda_us=round(event.device_time_total, 1),
                )
                for event in prof.key_averages()
                if event.device_time_total > 0
            ),
            key=lambda kernel: -kernel.cuda_us,
        )
        profiles.append(
            KvHeadsProfile(
                kv_heads=kv_heads,
                step_ms_p50=round(step_ms[TIMED_STEPS // 2], 3),
                kernels=kernels,
            )
        )

    return KernelReport(
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        batch_size=batch,
        seq_len=seq,
        profiles=profiles,
    ).model_dump(mode="json")


def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_BATCH, help="Batch size to profile"
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=DEFAULT_SEQ,
        help="Cached tokens per sequence at the profiled decode step",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="JSON report path (never overwritten)"
    )

    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output = Path(SCRIPT_DIR / args.output)
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output")

    with modal.enable_output(), app.run():
        report = KernelReport.model_validate(
            profile.remote(batch=args.batch, seq=args.seq)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    for kv_profile in report.profiles:
        logger.info(
            "kv_heads=%s step=%.3f ms", kv_profile.kv_heads, kv_profile.step_ms_p50
        )
        for kernel in kv_profile.kernels[:4]:
            logger.info("  %8.1f us x%d %s", kernel.cuda_us, kernel.count, kernel.name)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
