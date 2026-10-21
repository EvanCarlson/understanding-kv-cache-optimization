"""Generate the eager vs CUDA-graph decode-step chart for the GQA/MQA post.

Two panels on a shared y-axis, from a graph-decode report: the same decode step
timed eagerly (launch-bound) and replayed as a CUDA graph (launch overhead
removed). Graph bars for GQA and MQA carry their change versus MHA, and every bar has
a 95% bootstrap interval (over trials) as a whisker.

Run from this directory:
    uv run python generate_graph_decode_chart.py
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
from report_intervals import (
    GraphKey,
    graph_intervals,
    graph_percent_faster_than_mha,
)

DEFAULT_REPORT = "results/graph-decode-a100-2026-10-20.json"
DEFAULT_OUTPUT = "../../../static/images/charts/gqa-graph-decode.png"
# (batch_size, context_length) pairs on the x-axis, ordered by batch x context.
WORKLOADS = [(1, 1000), (128, 256), (64, 1000), (128, 512), (128, 1000)]
ARCHITECTURES = {12: "MHA (12 KV heads)", 3: "GQA (3 KV heads)", 1: "MQA (1 KV head)"}
ARCHITECTURE_COLORS = {12: COLOR_SERIES_1, 3: COLOR_SERIES_2, 1: COLOR_SERIES_3}
BAR_WIDTH = 0.26


def _percent_change_label(*, faster: Interval) -> str:
    return f"{-faster.estimate:+.0f}%"


def _plot_panel(
    *,
    ax: plt.Axes,
    intervals: dict[GraphKey, Interval],
    faster: dict[GraphKey, Interval],
    mode: str,
) -> None:
    for offset, heads in zip((-1, 0, 1), ARCHITECTURES):
        found = []
        xs = []
        for position, (batch_size, context_length) in enumerate(WORKLOADS):
            key = GraphKey(mode, heads, batch_size, context_length)
            interval = intervals[key]
            x = position + offset * BAR_WIDTH
            xs.append(x)
            found.append(interval)
            if mode == "graph" and heads != 12:
                ax.text(
                    x,
                    interval.high + 0.15,
                    _percent_change_label(faster=faster[key]),
                    ha="center",
                    va="bottom",
                    fontsize=7,
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
        [f"batch {b}\n{c:,} tok" for b, c in WORKLOADS],
        fontsize=8,
    )


def make_figure(*, report_path: Path) -> plt.Figure:
    """Create and return the eager vs graph figure. Does not save or display."""
    report = json.loads(report_path.read_text())
    intervals = graph_intervals(report=report)
    faster = graph_percent_faster_than_mha(report=report)
    fig, (ax_eager, ax_graph) = plt.subplots(
        1, 2, figsize=(12, 4.8), sharey=True, facecolor=COLOR_SURFACE
    )
    _plot_panel(ax=ax_eager, intervals=intervals, faster=faster, mode="eager")
    _plot_panel(ax=ax_graph, intervals=intervals, faster=faster, mode="graph")
    style_axis(
        ax=ax_eager,
        title="Eager: launch overhead sets the floor",
        ylabel="p50 decode step (ms)",
        xlabel="Workload",
    )
    style_axis(
        ax=ax_graph,
        title="CUDA graph: launch overhead removed",
        ylabel=None,
        xlabel="Workload",
    )
    ax_eager.set_ylim(bottom=0)
    ax_eager.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_MUTED)
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
        help="Graph-decode report JSON (default: %(default)s)",
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
