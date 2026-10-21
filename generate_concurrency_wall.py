"""Generate the concurrency-scaling chart for the KV cache optimization post.

Plots decode latency (ms, log scale) and peak tensor memory against batch size at
context_length=256 for the `basic` and `none` cache types, from a baseline sweep
report. Shows the latency gap between a KV cache and full recomputation widening
with concurrent batch size, and the memory the cache costs in return.

Run from this directory:
    uv run python generate_concurrency_wall.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from benchmark import CacheBenchmark
from chart_style import (
    COLOR_SERIES_1,
    COLOR_SERIES_2,
    COLOR_SURFACE,
    COLOR_TEXT_MUTED,
    DEFAULT_DPI,
    interval_note,
    plot_series,
    style_axis,
)
from confidence_intervals import Interval
from report_intervals import baseline_interval

DEFAULT_REPORT = "results/baseline-a100-2026-10-20-grouped.json"
DEFAULT_OUTPUT = "../../../static/images/charts/kv-cache-concurrency-wall.png"
CHART_CONTEXT_LENGTH = 256
GIB = 2**30


@dataclass(frozen=True)
class ConcurrencySeries:
    batch_sizes: list[int]
    p50_ms: list[float]
    p50_intervals: list[Interval]
    peak_gib: list[float]


def load_series(*, report: dict, cache_type: str) -> ConcurrencySeries:
    benchmark = CacheBenchmark.model_validate(report[cache_type])
    points = sorted(
        (
            p
            for p in benchmark.points
            if p.context_length == CHART_CONTEXT_LENGTH and p.decode_latency
        ),
        key=lambda p: p.batch_size,
    )
    if not points:
        raise ValueError(
            f"no {cache_type} points at context_length={CHART_CONTEXT_LENGTH}"
        )

    return ConcurrencySeries(
        batch_sizes=[p.batch_size for p in points],
        p50_ms=[p.decode_latency.p50_ms for p in points if p.decode_latency],
        p50_intervals=[baseline_interval(point=p) for p in points],
        peak_gib=[p.peak_memory_bytes / GIB for p in points],
    )


def make_figure(*, report_path: Path) -> plt.Figure:
    """Create and return the concurrency-wall figure. Does not save or display."""
    report = json.loads(report_path.read_text())
    basic = load_series(report=report, cache_type="basic")
    none = load_series(report=report, cache_type="none")

    fig, (ax_latency, ax_memory) = plt.subplots(
        1, 2, figsize=(12, 4.8), facecolor=COLOR_SURFACE
    )
    for ax in (ax_latency, ax_memory):
        ax.set_xscale("log", base=2)
        ax.set_xticks(basic.batch_sizes)
        ax.set_xticklabels([str(b) for b in basic.batch_sizes])
        ax.minorticks_off()

    for series, color, label in (
        (basic, COLOR_SERIES_1, "basic (KV cache)"),
        (none, COLOR_SERIES_2, "none (no cache)"),
    ):
        plot_series(
            ax=ax_latency,
            xs=series.batch_sizes,
            ys=series.p50_ms,
            color=color,
            label=label,
            intervals=series.p50_intervals,
        )
        plot_series(
            ax=ax_memory,
            xs=series.batch_sizes,
            ys=series.peak_gib,
            color=color,
            label=label,
        )

    ax_latency.set_yscale("log")
    style_axis(
        ax=ax_latency,
        title=f"Decode latency at context length {CHART_CONTEXT_LENGTH}",
        ylabel="p50 decode step (ms, log scale)",
        xlabel="Batch size",
    )
    style_axis(
        ax=ax_memory,
        title="Peak tensor memory",
        ylabel="GiB",
        xlabel="Batch size",
    )
    ax_latency.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_MUTED)
    ax_memory.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_MUTED)
    fig.tight_layout()
    fig.text(
        0.99,
        0.005,
        interval_note(intervals=basic.p50_intervals + none.p50_intervals),
        ha="right",
        fontsize=8,
        color=COLOR_TEXT_MUTED,
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="Baseline sweep report JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output PNG path (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output resolution in DPI (default: %(default)s)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = make_figure(report_path=Path(args.report))
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor=COLOR_SURFACE)
    plt.close(fig)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
