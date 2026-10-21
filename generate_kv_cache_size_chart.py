"""Generate the KV-cache size chart for the GQA/MQA post.

Bars for MHA (12 KV heads), GQA (3), and MQA (1) at batch size 1 and a
1,000-token prompt: bytes in use after prefill (solid) next to the bytes the static
cache reserves for 1,024 positions (light). Reads a matched attention sweep report.

Run from this directory:
    uv run python generate_kv_cache_size_chart.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pydantic import BaseModel, ConfigDict

from chart_style import (
    COLOR_SERIES_1,
    COLOR_SERIES_2,
    COLOR_SERIES_3,
    COLOR_SURFACE,
    COLOR_TEXT_MUTED,
    DEFAULT_DPI,
    style_axis,
)

DEFAULT_REPORT = "results/gqa-fp32-a100-2026-10-06.json"
DEFAULT_OUTPUT = "../../../static/images/charts/gqa-kv-cache-size.png"
CHART_BATCH_SIZE = 1
CHART_CONTEXT_LENGTH = 1000
ARCHITECTURES = {12: "MHA (12 KV heads)", 3: "GQA (3 KV heads)", 1: "MQA (1 KV head)"}
ARCHITECTURE_COLORS = {12: COLOR_SERIES_1, 3: COLOR_SERIES_2, 1: COLOR_SERIES_3}
BAR_WIDTH = 0.36
ALLOCATED_ALPHA = 0.35
MIB = 2**20


class SweepRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    heads: int
    context_length: int
    batch_size: int
    cache_bytes_used: int
    cache_bytes_allocated: int


@dataclass(frozen=True)
class CacheSize:
    used_mib: float
    allocated_mib: float


def load_cache_sizes(*, report: dict) -> dict[int, CacheSize]:
    """Cache bytes after prefill for each KV-head count at the chart workload.

    Cache bytes do not depend on the attention implementation, so the first
    matching run per head count is enough.
    """
    sizes: dict[int, CacheSize] = {}
    for raw in report["runs"]:
        run = SweepRun.model_validate(raw)
        if (
            run.batch_size == CHART_BATCH_SIZE
            and run.context_length == CHART_CONTEXT_LENGTH
            and run.heads not in sizes
        ):
            sizes[run.heads] = CacheSize(
                used_mib=run.cache_bytes_used / MIB,
                allocated_mib=run.cache_bytes_allocated / MIB,
            )
    missing = set(ARCHITECTURES) - set(sizes)
    if missing:
        raise ValueError(
            f"no runs for KV heads {sorted(missing)} at the chart workload"
        )

    return sizes


def make_figure(*, report_path: Path) -> plt.Figure:
    """Create and return the KV-cache size figure. Does not save or display."""
    sizes = load_cache_sizes(report=json.loads(report_path.read_text()))
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=COLOR_SURFACE)

    mha_used = sizes[12].used_mib
    for position, heads in enumerate(ARCHITECTURES):
        size = sizes[heads]
        color = ARCHITECTURE_COLORS[heads]
        for offset, value, alpha in (
            (-0.5, size.used_mib, 1.0),
            (0.5, size.allocated_mib, ALLOCATED_ALPHA),
        ):
            x = position + offset * BAR_WIDTH
            ax.bar(
                x,
                value,
                width=BAR_WIDTH,
                color=color,
                alpha=alpha,
                edgecolor=COLOR_SURFACE,
                linewidth=1,
                zorder=3,
            )
            ax.text(
                x,
                value + 1.2,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLOR_TEXT_MUTED,
            )
        if heads != 12:
            ax.text(
                position,
                -9,
                f"{mha_used / size.used_mib:.0f}x smaller than MHA",
                ha="center",
                va="top",
                fontsize=9,
                color=COLOR_TEXT_MUTED,
            )

    ax.set_xticks(range(len(ARCHITECTURES)))
    ax.set_xticklabels(list(ARCHITECTURES.values()), fontsize=10)
    ax.set_ylim(0, sizes[12].allocated_mib * 1.15)
    style_axis(
        ax=ax,
        title=(
            f"KV cache at batch size {CHART_BATCH_SIZE} and "
            f"{CHART_CONTEXT_LENGTH:,} tokens"
        ),
        ylabel="MiB",
        xlabel="",
    )
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_TEXT_MUTED, label="In use after prefill"),
            Patch(
                facecolor=COLOR_TEXT_MUTED,
                alpha=ALLOCATED_ALPHA,
                label="Reserved for 1,024 positions",
            ),
        ],
        loc="upper right",
        frameon=False,
        labelcolor=COLOR_TEXT_MUTED,
    )
    fig.tight_layout()

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
