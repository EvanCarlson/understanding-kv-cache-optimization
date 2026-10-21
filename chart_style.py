"""Shared chart style for the KV-cache series: validated categorical palette and axis cleanup.

Series colors are the first 3 slots of a palette validated for color-vision
deficiency (adjacent and all-pairs). The third slot (aqua) is below 3:1 contrast
on white, so every chart ships a legend and its post ships a full-numbers table.
"""

import matplotlib.pyplot as plt

from confidence_intervals import Interval

COLOR_SERIES_1 = "#2a78d6"
COLOR_SERIES_2 = "#eb6834"
COLOR_SERIES_3 = "#1baf7a"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#d9d8d2"
COLOR_SURFACE = "#ffffff"
DEFAULT_DPI = 150
# Whiskers narrower than this share of the estimate vanish behind the markers.
NARROW_INTERVAL_FRACTION = 0.03


def style_axis(*, ax: plt.Axes, title: str, ylabel: str | None, xlabel: str) -> None:
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(True, axis="y", which="major", linewidth=0.6, color=COLOR_GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=9)
    ax.set_title(title, fontsize=11, color=COLOR_TEXT, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=COLOR_TEXT_MUTED)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=COLOR_TEXT_MUTED)


def plot_series(
    *,
    ax: plt.Axes,
    xs: list[int],
    ys: list[float],
    color: str,
    label: str | None = None,
    intervals: list[Interval] | None = None,
) -> None:
    """Line with markers; whiskers show each point's interval when given."""
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2,
        marker="o",
        markersize=7,
        markeredgecolor=COLOR_SURFACE,
        markeredgewidth=1.5,
        zorder=3,
        label=label,
    )
    if intervals is not None:
        ax.errorbar(
            xs,
            ys,
            yerr=[
                [y - i.low for y, i in zip(ys, intervals)],
                [i.high - y for y, i in zip(ys, intervals)],
            ],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            zorder=2,
        )


def interval_note(*, intervals: list[Interval]) -> str:
    """Footnote for whiskered charts; says so when the whiskers are too narrow to see."""
    note = "Latency whiskers: 95% bootstrap interval over trials"
    widest = max((i.high - i.low) / i.estimate for i in intervals)
    if widest < NARROW_INTERVAL_FRACTION:
        note += " (too narrow to see at this scale)"

    return note
