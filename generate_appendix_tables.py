"""Print the posts' appendix tables, with 95% bootstrap intervals, as Markdown.

Reads one saved report and prints its tables to stdout so the appendices can be
pasted into a post without retyping numbers. Choose the report kind with --kind:
`baseline` (cached versus uncached sweep, Post 1), `sweep` (eager attention sweep,
Post 2), or `graph` (CUDA-graph experiment, Post 2).

Run from this directory:
    uv run python generate_appendix_tables.py --kind sweep --report results/<file>.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark import BenchmarkPoint, CacheBenchmark
from confidence_intervals import Interval
from report_intervals import (
    GraphKey,
    SweepKey,
    baseline_interval,
    baseline_speedup_interval,
    graph_intervals,
    graph_percent_faster_than_mha,
    sweep_intervals,
    sweep_percent_faster_than_mha,
    sweep_rounds,
)

DEFAULT_KIND = "baseline"
KINDS = ["baseline", "sweep", "graph"]
# (batch_size, context_length), in the order the posts' charts use.
WORKLOADS = [(1, 1000), (128, 256), (64, 1000), (128, 512), (128, 1000)]
SWEEP_STAGES = {
    "grouped_fp32": "Grouped",
    "expanded_fp32": "Expanded",
    "repeat_kv_fp32": "`repeat_kv`",
}
ARCHITECTURES = {12: "MHA", 3: "GQA", 1: "MQA"}
MIB = 2**20
GIB = 2**30


def _interval_cell(*, interval: Interval) -> str:
    return f"{interval.estimate:.2f} ({interval.low:.2f} to {interval.high:.2f})"


def _percent_cell(*, interval: Interval) -> str:
    return f"{interval.estimate:+.1f}% ({interval.low:+.1f} to {interval.high:+.1f})"


def _table(*, header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)

    return "\n".join(lines)


def _bytes_cell(*, value: int) -> str:
    return f"{value / MIB:.1f} MiB"


def _points(*, report: dict, cache_type: str) -> dict[tuple[int, int], BenchmarkPoint]:
    benchmark = CacheBenchmark.model_validate(report[cache_type])

    return {(p.batch_size, p.context_length): p for p in benchmark.points}


def baseline_tables(*, report: dict) -> str:
    basic = _points(report=report, cache_type="basic")
    none = _points(report=report, cache_type="none")

    context_rows = []
    for name, points in (
        ("No cache (`none`)", none),
        ("Basic KV cache (`basic`)", basic),
    ):
        for (batch, context), point in sorted(points.items(), key=lambda kv: kv[0][1]):
            if batch != 1 or point.decode_latency is None:
                continue
            latency = point.decode_latency
            context_rows.append(
                [
                    name,
                    f"{context:,}",
                    _interval_cell(interval=baseline_interval(point=point)),
                    f"{latency.p95_ms:.2f}",
                    f"{latency.p99_ms:.2f}",
                    _bytes_cell(value=point.cache_bytes_used),
                ]
            )

    batch_rows = []
    for (batch, context), cached in sorted(basic.items()):
        if context != 256:
            continue
        uncached = none[(batch, context)]
        if cached.decode_latency is None or uncached.decode_latency is None:
            continue
        speedup = baseline_speedup_interval(none=uncached, basic=cached)
        batch_rows.append(
            [
                str(batch),
                _interval_cell(interval=baseline_interval(point=cached)),
                f"{cached.decode_latency.p95_ms:.2f} / {cached.decode_latency.p99_ms:.2f}",
                _interval_cell(interval=baseline_interval(point=uncached)),
                f"{uncached.decode_latency.p95_ms:.2f} / {uncached.decode_latency.p99_ms:.2f}",
                f"{speedup.estimate:.2f}x ({speedup.low:.2f} to {speedup.high:.2f})",
                f"{cached.peak_memory_bytes / GIB:.2f} GiB",
                f"{uncached.peak_memory_bytes / GIB:.2f} GiB",
            ]
        )

    return "\n\n".join(
        [
            "### Context-Length Sweep (Batch Size 1)",
            "Latencies are p50 in ms with a 95% bootstrap interval; p95 and p99 are point estimates.",
            _table(
                header=[
                    "Cache type",
                    "Context length",
                    "p50 (ms, 95% interval)",
                    "p95 (ms)",
                    "p99 (ms)",
                    "Cache bytes used",
                ],
                rows=context_rows,
            ),
            "### Batch-Size Sweep (Context Length 256)",
            _table(
                header=[
                    "Batch size",
                    "`basic` p50 (ms, 95% interval)",
                    "`basic` p95 / p99 (ms)",
                    "`none` p50 (ms, 95% interval)",
                    "`none` p95 / p99 (ms)",
                    "Speedup, `none` over `basic` (95% interval)",
                    "`basic` peak memory",
                    "`none` peak memory",
                ],
                rows=batch_rows,
            ),
        ]
    )


def _latency_percentile_cell(
    *, report: dict, key: SweepKey, percentile: float
) -> float:
    rounds = sweep_rounds(
        report=report, workloads=[(key.batch_size, key.context_length)]
    )
    return float(
        np.mean(
            [
                np.percentile(np.concatenate(trials), percentile)
                for trials in rounds[key]
            ]
        )
    )


def sweep_tables(*, report: dict) -> str:
    intervals = sweep_intervals(report=report, workloads=WORKLOADS)
    faster = sweep_percent_faster_than_mha(report=report, workloads=WORKLOADS)

    percentile_rows = []
    interval_rows = []
    change_rows = []
    for stage, name in SWEEP_STAGES.items():
        for batch, context in WORKLOADS:
            lead = [name, str(batch), f"{context:,}"]
            percentile_cells = []
            interval_cells = []
            change_cells = []
            for heads in ARCHITECTURES:
                key = SweepKey(stage, heads, batch, context)
                percentiles = [
                    _latency_percentile_cell(report=report, key=key, percentile=q)
                    for q in (50, 95, 99)
                ]
                percentile_cells.append(" / ".join(f"{v:.2f}" for v in percentiles))
                interval_cells.append(_interval_cell(interval=intervals[key]))
                if heads != 12:
                    change_cells.append(_percent_cell(interval=faster[key]))
            percentile_rows.append(lead + percentile_cells)
            interval_rows.append(lead + interval_cells)
            change_rows.append(lead + change_cells)

    lead_header = ["Implementation", "Batch size", "Context length"]

    return "\n\n".join(
        [
            "### Eager Sweep",
            "Each cell lists p50 / p95 / p99 step latency in milliseconds.",
            _table(
                header=lead_header
                + [f"{a} p50 / p95 / p99 (ms)" for a in ARCHITECTURES.values()],
                rows=percentile_rows,
            ),
            "### Eager Sweep: p50 With 95% Intervals",
            (
                "Intervals come from resampling trials; they describe noise within one "
                "allocation, not differences between machines."
            ),
            _table(
                header=lead_header
                + [f"{a} p50 (ms, 95% interval)" for a in ARCHITECTURES.values()],
                rows=interval_rows,
            ),
            "### Eager Sweep: Change vs. MHA",
            (
                "Percent faster than MHA in the same implementation and workload; "
                "negative means slower."
            ),
            _table(
                header=lead_header
                + [f"{a} vs. MHA (95% interval)" for a in ("GQA", "MQA")],
                rows=change_rows,
            ),
        ]
    )


def graph_tables(*, report: dict) -> str:
    intervals = graph_intervals(report=report)
    faster = graph_percent_faster_than_mha(report=report)

    p50_rows = []
    for batch, context in WORKLOADS:
        for mode, label in (("eager", "Eager"), ("graph", "CUDA graph")):
            p50_rows.append(
                [str(batch), f"{context:,}", label]
                + [
                    _interval_cell(
                        interval=intervals[GraphKey(mode, h, batch, context)]
                    )
                    for h in ARCHITECTURES
                ]
            )
    change_rows = [
        [str(batch), f"{context:,}"]
        + [
            _percent_cell(interval=faster[GraphKey("graph", h, batch, context)])
            for h in (3, 1)
        ]
        for batch, context in WORKLOADS
    ]

    return "\n\n".join(
        [
            "### CUDA-Graph Decode Experiment",
            _table(
                header=["Batch size", "Context length", "Mode"]
                + [f"{a} p50 (ms, 95% interval)" for a in ARCHITECTURES.values()],
                rows=p50_rows,
            ),
            "### CUDA-Graph Decode: Change vs. MHA",
            "Percent faster than MHA for the replayed graph; negative means slower.",
            _table(
                header=["Batch size", "Context length"]
                + [f"{a} vs. MHA (95% interval)" for a in ("GQA", "MQA")],
                rows=change_rows,
            ),
        ]
    )


def render(*, kind: str, report: dict) -> str:
    if kind == "baseline":
        return baseline_tables(report=report)
    if kind == "sweep":
        return sweep_tables(report=report)

    return graph_tables(report=report)


def parse_args(*, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kind", choices=KINDS, default=DEFAULT_KIND)
    parser.add_argument("--report", required=True, help="Report JSON to summarize")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.report).read_text())
    print(render(kind=args.kind, report=report))


if __name__ == "__main__":
    main()
