"""Generate the GQA/MQA decode-latency chart from a matched attention sweep report.

One panel per attention implementation (grouped, expanded, repeat_kv) on a shared
y-axis, with MHA, GQA, and MQA bars for each workload. Each bar is the median of
the two round-level p50s, as in the post's tables, with a 95% bootstrap interval
(over trials) as the whisker. Workloads missing a bar (an OOM) are labeled instead
of dropped.

Run from this directory:
    uv run python generate_gqa_charts.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from chart_style import (
    COLOR_SERIES_1,
    COLOR_SERIES_2,
    COLOR_SERIES_3,
    COLOR_SURFACE,
    COLOR_TEXT_MUTED,
    DEFAULT_DPI,
    style_axis,
)
from confidence_intervals import Interval
from report_intervals import SweepKey, sweep_intervals

DEFAULT_REPORT = "results/gqa-fp32-a100-2026-10-06.json"
DEFAULT_OUTPUT = "../../../static/images/charts/gqa-decode-latency.png"
# (batch_size, context_length) pairs on the x-axis, ordered by batch x context.
WORKLOADS = [(1, 1000), (128, 256), (64, 1000), (128, 512), (128, 1000)]
ARCHITECTURES = {12: "MHA (12 KV heads)", 3: "GQA (3 KV heads)", 1: "MQA (1 KV head)"}
ARCHITECTURE_COLORS = {12: COLOR_SERIES_1, 3: COLOR_SERIES_2, 1: COLOR_SERIES_3}
PANELS = {
    "grouped_fp32": "Grouped: K/V stay compact",
    "expanded_fp32": "Expanded: copies K/V, even for MHA",
    "repeat_kv_fp32": "repeat_kv: copies K/V, skips MHA",
}
BAR_WIDTH = 0.26


def _workload_label(*, batch_size: int, context_length: int) -> str:
    return f"batch {batch_size}\n{context_length:,} tok"


def _plot_panel(
    *, ax: plt.Axes, stage: str, intervals: dict[SweepKey, Interval], ymax: float
) -> None:
    for offset, heads in zip((-1, 0, 1), ARCHITECTURES):
        xs: list[float] = []
        found: list[Interval] = []
        for position, (batch_size, context_length) in enumerate(WORKLOADS):
            key = SweepKey(stage, heads, batch_size, context_length)
            x = position + offset * BAR_WIDTH
            if key in intervals:
                xs.append(x)
                found.append(intervals[key])
            else:
                ax.text(
                    x,
                    ymax * 0.02,
                    "OOM",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=COLOR_TEXT_MUTED,
                )
        ax.bar(
            xs,
            [i.estimate for i in found],
            yerr=[
                [i.estimate - i.low for i in found],
                [i.high - i.estimate for i in found],
            ],
            error_kw={"ecolor": COLOR_TEXT_MUTED, "elinewidth": 1, "capsize": 2},
            width=BAR_WIDTH,
            color=ARCHITECTURE_COLORS[heads],
            edgecolor=COLOR_SURFACE,
            linewidth=1,
            label=ARCHITECTURES[heads],
            zorder=3,
        )

    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels(
        [_workload_label(batch_size=b, context_length=c) for b, c in WORKLOADS],
        fontsize=8,
    )
    ax.set_ylim(0, ymax)


def make_figure(*, report_path: Path) -> plt.Figure:
    """Create and return the three-panel latency figure. Does not save or display."""
    intervals = sweep_intervals(
        report=json.loads(report_path.read_text()), workloads=WORKLOADS
    )
    ymax = max(i.high for i in intervals.values()) * 1.08

    fig, axes = plt.subplots(
        1, len(PANELS), figsize=(15, 4.8), sharey=True, facecolor=COLOR_SURFACE
    )
    for index, (ax, (stage, title)) in enumerate(zip(axes, PANELS.items())):
        _plot_panel(ax=ax, stage=stage, intervals=intervals, ymax=ymax)
        style_axis(
            ax=ax,
            title=title,
            ylabel="p50 decode step (ms)" if index == 0 else None,
            xlabel="Workload",
        )
    axes[0].legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_MUTED)
    fig.tight_layout()
    fig.text(
        0.99,
        0.005,
        "Whiskers: 95% bootstrap interval over trials",
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
        help="Matched attention sweep report JSON (default: %(default)s)",
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
