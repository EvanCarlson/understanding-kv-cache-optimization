"""Time one GPT-2-small decode step eagerly and as a replayed CUDA graph on an A100.

Eager steps pay Python dispatch and kernel-launch overhead for every small kernel.
A graph replay removes it, leaving GPU time, where reading less KV cache can
actually shorten the step. Each cell prefills a prompt, times eager steps, then
captures one decode step at a fixed cache length and times replays of it. Weights
are random: timing depends on shapes, not values. Uses grouped attention, so K/V
stay compact. Writes a JSON report and launches a billed Modal A100 job.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import modal
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from model import KVCacheFactory, ModelConfig, TinyTransformer

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = "results/graph-decode-a100-2026-10-07.json"
DEFAULT_ROUNDS = 2
DEFAULT_TRIALS = 5
DEFAULT_EAGER_STEPS = 10
DEFAULT_GRAPH_REPLAYS = 50
DEFAULT_SEED = 42
KV_HEADS = [12, 3, 1]
# (batch_size, context_length), matching the GQA post's workloads.
POINTS = [(1, 1000), (128, 256), (64, 1000), (128, 512), (128, 1000)]
WARMUP_STEPS = 2
MAX_POSITIONS = 1024
GPT2_SMALL_VOCAB = 50257
app = modal.App("gqa-graph-decode")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(SCRIPT_DIR))
    .workdir("/")
    .add_local_python_source("model")
)


class GraphCell(BaseModel):
    model_config = ConfigDict(frozen=True)
    round: int
    kv_heads: int
    batch_size: int
    context_length: int
    # Medians over every step of every trial; the raw lists below let confidence
    # intervals resample whole trials (one fresh prompt and graph capture each).
    eager_p50_ms: float
    graph_p50_ms: float
    eager_ms: list[list[float]]
    graph_ms: list[list[float]]


class GraphReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    gpu: str
    torch: str
    seed: int
    trials: int
    eager_steps: int
    graph_replays: int
    cells: list[GraphCell]


@dataclass(frozen=True)
class TrialTimings:
    eager_ms: list[float]
    graph_ms: list[float]


def _timed_ms(*, step) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    step()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000


def _measure_cell(
    *,
    model: TinyTransformer,
    config: ModelConfig,
    batch: int,
    context: int,
    eager_steps: int,
    graph_replays: int,
) -> TrialTimings:
    cache = KVCacheFactory.create(
        "gqa", config=config, batch_size=batch, device=torch.device("cuda")
    )
    prompt = torch.randint(0, config.vocab_size, (batch, context), device="cuda")
    token = torch.randint(0, config.vocab_size, (batch, 1), device="cuda")

    def decode() -> None:
        model(token, cache=cache)

    with torch.no_grad():
        model(prompt, cache=cache, logits_to_keep=1)
        for _ in range(WARMUP_STEPS):
            decode()
        eager = [_timed_ms(step=decode) for _ in range(eager_steps)]

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP_STEPS):
                decode()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decode()
        replays = [_timed_ms(step=graph.replay) for _ in range(graph_replays)]

    return TrialTimings(eager_ms=eager, graph_ms=replays)


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def experiment(
    *, rounds: int, trials: int, eager_steps: int, graph_replays: int, seed: int
) -> dict:
    cells = []
    for round_idx in range(rounds):
        heads_order = KV_HEADS if round_idx % 2 == 0 else list(reversed(KV_HEADS))
        for kv_heads in heads_order:
            torch.manual_seed(seed)
            config = ModelConfig(
                vocab_size=GPT2_SMALL_VOCAB,
                d_model=768,
                max_seq_len=MAX_POSITIONS,
                n_layers=12,
                num_heads=12,
                num_kv_heads=kv_heads,
                attention_backend="grouped",
            )
            model = TinyTransformer(config=config).to("cuda").eval()
            for batch, context in POINTS:
                timings = [
                    _measure_cell(
                        model=model,
                        config=config,
                        batch=batch,
                        context=context,
                        eager_steps=eager_steps,
                        graph_replays=graph_replays,
                    )
                    for _ in range(trials)
                ]
                eager_ms = [t.eager_ms for t in timings]
                graph_ms = [t.graph_ms for t in timings]
                cells.append(
                    GraphCell(
                        round=round_idx + 1,
                        kv_heads=kv_heads,
                        batch_size=batch,
                        context_length=context,
                        eager_p50_ms=float(np.percentile(np.concatenate(eager_ms), 50)),
                        graph_p50_ms=float(np.percentile(np.concatenate(graph_ms), 50)),
                        eager_ms=eager_ms,
                        graph_ms=graph_ms,
                    )
                )
                logger.info(
                    "%s", cells[-1].model_dump_json(exclude={"eager_ms", "graph_ms"})
                )
            del model
            torch.cuda.empty_cache()

    return GraphReport(
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        seed=seed,
        trials=trials,
        eager_steps=eager_steps,
        graph_replays=graph_replays,
        cells=cells,
    ).model_dump(mode="json")


def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Independent prompts and graph captures per cell and round",
    )
    parser.add_argument("--eager-steps", type=int, default=DEFAULT_EAGER_STEPS)
    parser.add_argument("--graph-replays", type=int, default=DEFAULT_GRAPH_REPLAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="JSON report path (never overwritten)"
    )
    args = parser.parse_args(argv)
    if min(args.rounds, args.trials, args.eager_steps, args.graph_replays) < 1:
        parser.error("Rounds, trials, eager steps, and graph replays must be positive")
    context = max(context for _, context in POINTS)
    if context + 1 + 2 * WARMUP_STEPS + args.eager_steps + 1 > MAX_POSITIONS:
        parser.error(
            "Prefill, warmup, eager, and captured steps must fit 1024 positions"
        )

    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output = SCRIPT_DIR / args.output
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new --output")

    with modal.enable_output(), app.run(detach=True):
        report = GraphReport.model_validate(
            experiment.remote(
                rounds=args.rounds,
                trials=args.trials,
                eager_steps=args.eager_steps,
                graph_replays=args.graph_replays,
                seed=args.seed,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
