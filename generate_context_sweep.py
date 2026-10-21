"""Generate the context-length sweep chart for the KV cache optimization post.

Plots decode latency (p50) for the `basic` and `none` cache types against context
length at batch size 1, next to the bytes the `basic` cache holds after prefill.
Reads a baseline sweep report.

Run from this directory:
    uv run python generate_context_sweep.py
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
DEFAULT_OUTPUT = "../../../static/images/charts/kv-cache-context-sweep.png"
CHART_BATCH_SIZE = 1
MIB = 2**20


@dataclass(frozen=True)
class ContextSeries:
    context_lengths: list[int]
    p50_ms: list[float]
    p50_intervals: list[Interval]
    cache_used_mib: list[float]


def load_series(*, report: dict, cache_type: str) -> ContextSeries:
    benchmark = CacheBenchmark.model_validate(report[cache_type])
    points = sorted(
        (
            p
            for p in benchmark.points
            if p.batch_size == CHART_BATCH_SIZE and p.decode_latency
        ),
        key=lambda p: p.context_length,
    )
    if not points:
        raise ValueError(f"no {cache_type} points at batch_size={CHART_BATCH_SIZE}")

    return ContextSeries(
        context_lengths=[p.context_length for p in points],
        p50_ms=[p.decode_latency.p50_ms for p in points if p.decode_latency],
        p50_intervals=[baseline_interval(point=p) for p in points],
        cache_used_mib=[p.cache_bytes_used / MIB for p in points],
    )


def make_figure(*, report_path: Path) -> plt.Figure:
    """Create and return the context-sweep figure. Does not save or display."""
    report = json.loads(report_path.read_text())
    basic = load_series(report=report, cache_type="basic")
    none = load_series(report=report, cache_type="none")

    fig, (ax_latency, ax_cache) = plt.subplots(
        1, 2, figsize=(12, 4.8), facecolor=COLOR_SURFACE
    )
    for series, color, label in (
        (basic, COLOR_SERIES_1, "basic (KV cache)"),
        (none, COLOR_SERIES_2, "none (no cache)"),
    ):
        plot_series(
            ax=ax_latency,
            xs=series.context_lengths,
            ys=series.p50_ms,
            color=color,
            label=label,
            intervals=series.p50_intervals,
        )
    plot_series(
        ax=ax_cache,
        xs=basic.context_lengths,
        ys=basic.cache_used_mib,
        color=COLOR_SERIES_1,
    )

    for ax in (ax_latency, ax_cache):
        ax.set_xticks(basic.context_lengths)
        ax.set_xticklabels([f"{c:,}" for c in basic.context_lengths])
    ax_latency.set_ylim(bottom=0)
    ax_cache.set_ylim(bottom=0)

    style_axis(
        ax=ax_latency,
        title=f"Decode latency at batch size {CHART_BATCH_SIZE}",
        ylabel="p50 decode step (ms)",
        xlabel="Context length (tokens)",
    )
    style_axis(
        ax=ax_cache,
        title="Cache bytes in use after prefill (basic)",
        ylabel="MiB",
        xlabel="Context length (tokens)",
    )
    ax_latency.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_MUTED)
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
