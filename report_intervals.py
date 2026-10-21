"""Turn saved benchmark reports into confidence intervals for the posts' charts and tables.

Three report formats feed the posts: the cached-versus-uncached baseline sweep
(`benchmark_baseline.py`), the eager attention sweep (`benchmark_optimizations.py`),
and the CUDA-graph experiment (`benchmark_graph_decode.py`). Each adapter groups raw
step latencies into trials and rounds, then calls `confidence_intervals`.
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from benchmark import BenchmarkPoint
from confidence_intervals import (
    Interval,
    Trials,
    median_interval,
    percent_faster,
    ratio_interval,
    split_into_trials,
)

MHA_HEADS = 12
Workload = tuple[int, int]  # (batch_size, context_length)


@dataclass(frozen=True)
class SweepKey:
    stage: str
    heads: int
    batch_size: int
    context_length: int


@dataclass(frozen=True)
class GraphKey:
    mode: str
    heads: int
    batch_size: int
    context_length: int


class SweepRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    round: int
    heads: int
    stage: str
    context_length: int
    batch_size: int
    latencies_ms: list[float]


class GraphCell(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    round: int
    kv_heads: int
    batch_size: int
    context_length: int
    eager_ms: list[list[float]]
    graph_ms: list[list[float]]


def baseline_interval(*, point: BenchmarkPoint) -> Interval:
    """Interval for one cached-or-uncached baseline point (a single round)."""
    return median_interval(rounds=[point.trial_latencies_ms])


def baseline_speedup_interval(
    *, none: BenchmarkPoint, basic: BenchmarkPoint
) -> Interval:
    """Interval for how many times slower the uncached step is than the cached one."""
    return ratio_interval(
        numerator=[none.trial_latencies_ms], denominator=[basic.trial_latencies_ms]
    )


def sweep_rounds(
    *, report: dict, workloads: list[Workload] | None = None
) -> dict[SweepKey, list[Trials]]:
    """Group eager-sweep runs by workload, one entry of trials per round.

    The reports hold every context/batch pair, and bootstrapping all of them is slow,
    so callers pass only the workloads they plot.
    """
    num_trials = report["metadata"]["trials"]
    grouped: dict[SweepKey, list[Trials]] = {}
    for raw in sorted(report["runs"], key=lambda r: r["round"]):
        run = SweepRun.model_validate(raw)
        if (
            workloads is not None
            and (run.batch_size, run.context_length) not in workloads
        ):
            continue
        key = SweepKey(run.stage, run.heads, run.batch_size, run.context_length)
        grouped.setdefault(key, []).append(
            split_into_trials(latencies_ms=run.latencies_ms, num_trials=num_trials)
        )

    return grouped


def sweep_intervals(
    *, report: dict, workloads: list[Workload] | None = None
) -> dict[SweepKey, Interval]:
    return {
        key: median_interval(rounds=rounds)
        for key, rounds in sweep_rounds(report=report, workloads=workloads).items()
    }


def sweep_percent_faster_than_mha(
    *, report: dict, workloads: list[Workload] | None = None
) -> dict[SweepKey, Interval]:
    """Percent faster than MHA at the same stage and workload; negative is slower."""
    rounds = sweep_rounds(report=report, workloads=workloads)

    return {
        key: percent_faster(
            ratio=ratio_interval(
                numerator=rounds[key],
                denominator=rounds[
                    SweepKey(key.stage, MHA_HEADS, key.batch_size, key.context_length)
                ],
            )
        )
        for key in rounds
        if key.heads != MHA_HEADS
    }


def graph_rounds(*, report: dict) -> dict[GraphKey, list[Trials]]:
    """Group graph-experiment cells by mode and workload, one entry per round."""
    grouped: dict[GraphKey, list[Trials]] = {}
    for raw in sorted(report["cells"], key=lambda c: c["round"]):
        cell = GraphCell.model_validate(raw)
        for mode, trials in (("eager", cell.eager_ms), ("graph", cell.graph_ms)):
            key = GraphKey(mode, cell.kv_heads, cell.batch_size, cell.context_length)
            grouped.setdefault(key, []).append(trials)

    return grouped


def graph_intervals(*, report: dict) -> dict[GraphKey, Interval]:
    return {
        key: median_interval(rounds=rounds)
        for key, rounds in graph_rounds(report=report).items()
    }


def graph_percent_faster_than_mha(*, report: dict) -> dict[GraphKey, Interval]:
    rounds = graph_rounds(report=report)

    return {
        key: percent_faster(
            ratio=ratio_interval(
                numerator=rounds[key],
                denominator=rounds[
                    GraphKey(key.mode, MHA_HEADS, key.batch_size, key.context_length)
                ],
            )
        )
        for key in rounds
        if key.heads != MHA_HEADS
    }
