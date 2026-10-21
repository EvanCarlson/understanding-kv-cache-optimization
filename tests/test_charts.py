from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import generate_concurrency_wall
import generate_context_sweep
import generate_gqa_charts
import generate_graph_decode_chart
import generate_kv_cache_size_chart
from chart_style import interval_note
from confidence_intervals import Interval
from report_intervals import (
    GraphKey,
    SweepKey,
    graph_intervals,
    graph_percent_faster_than_mha,
    sweep_intervals,
    sweep_percent_faster_than_mha,
)


def _point(*, context: int, batch: int, p50: float, peak: int, used: int) -> dict:
    return {
        "context_length": context,
        "batch_size": batch,
        "decode_latency": {"p50_ms": p50, "p95_ms": p50, "p99_ms": p50},
        "throughput_tokens_per_sec": 1.0,
        "num_trials": 1,
        "cache_bytes_used": used,
        "peak_memory_bytes": peak,
        "oom": False,
        "trial_latencies_ms": [[p50 - 0.5, p50 + 0.5], [p50 - 0.5, p50 + 0.5]],
    }


def _baseline_report() -> dict:
    points = {
        "basic": [
            _point(context=256, batch=8, p50=9.0, peak=2 * 2**30, used=2**20),
            _point(context=256, batch=1, p50=8.0, peak=2**30, used=2**20),
            _point(context=64, batch=1, p50=7.0, peak=2**30, used=2 * 2**20),
        ],
        "none": [
            _point(context=256, batch=1, p50=9.0, peak=2**29, used=0),
            _point(context=64, batch=1, p50=8.5, peak=2**29, used=0),
            _point(context=256, batch=8, p50=30.0, peak=2**30, used=0),
        ],
    }
    return {
        cache_type: {"cache_type": cache_type, "gpu_type": "test", "points": pts}
        for cache_type, pts in points.items()
    }


def _sweep_report() -> dict:
    def run(*, round: int, heads: int, stage: str, p50: float) -> dict:
        return {
            "round": round,
            "heads": heads,
            "stage": stage,
            "context_length": 256,
            "batch_size": 128,
            "p50_ms": p50,
            "latencies_ms": [p50 - 0.5, p50 + 0.5, p50 - 0.5, p50 + 0.5],
        }

    return {
        "metadata": {"trials": 2},
        "runs": [
            run(round=1, heads=12, stage="grouped_fp32", p50=8.0),
            run(round=2, heads=12, stage="grouped_fp32", p50=10.0),
            run(round=1, heads=3, stage="grouped_fp32", p50=9.0),
            run(round=2, heads=3, stage="grouped_fp32", p50=9.0),
        ],
    }


def test_sweep_intervals_estimate_is_mean_of_round_medians() -> None:
    intervals = sweep_intervals(report=_sweep_report())

    mha = intervals[SweepKey("grouped_fp32", 12, 128, 256)]
    assert mha.estimate == pytest.approx(9.0)
    assert mha.low <= mha.estimate <= mha.high


def test_sweep_percent_faster_than_mha_skips_mha_itself() -> None:
    faster = sweep_percent_faster_than_mha(report=_sweep_report())

    assert SweepKey("grouped_fp32", 12, 128, 256) not in faster
    assert faster[SweepKey("grouped_fp32", 3, 128, 256)].estimate == pytest.approx(0.0)


def test_sweep_intervals_filters_to_requested_workloads() -> None:
    assert sweep_intervals(report=_sweep_report(), workloads=[(1, 1000)]) == {}


def test_concurrency_series_sorted_by_batch_at_chart_context() -> None:
    series = generate_concurrency_wall.load_series(
        report=_baseline_report(), cache_type="basic"
    )
    assert series.batch_sizes == [1, 8]
    assert series.p50_ms == [8.0, 9.0]
    assert series.peak_gib == pytest.approx([1.0, 2.0])


def test_context_series_sorted_by_context_at_chart_batch() -> None:
    series = generate_context_sweep.load_series(
        report=_baseline_report(), cache_type="basic"
    )
    assert series.context_lengths == [64, 256]
    assert series.cache_used_mib == pytest.approx([2.0, 1.0])


def test_load_series_rejects_report_without_chart_points() -> None:
    report = _baseline_report()
    report["none"]["points"] = [
        _point(context=64, batch=1, p50=8.5, peak=2**29, used=0)
    ]
    with pytest.raises(ValueError, match="no none points"):
        generate_concurrency_wall.load_series(report=report, cache_type="none")


def test_figures_render_and_save(tmp_path: Path) -> None:
    import json

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_baseline_report()))
    sweep = tmp_path / "sweep.json"
    sweep.write_text(json.dumps(_sweep_report()))
    for fig in (
        generate_concurrency_wall.make_figure(report_path=baseline),
        generate_context_sweep.make_figure(report_path=baseline),
        generate_gqa_charts.make_figure(report_path=sweep),
    ):
        out = tmp_path / "chart.png"
        fig.savefig(out)
        assert out.stat().st_size > 0


def _graph_report() -> dict:
    cells = []
    for round_idx, scale in ((1, 1.0), (2, 1.2)):
        for heads in (12, 3, 1):
            for batch, context in generate_graph_decode_chart.WORKLOADS:
                cells.append(
                    {
                        "round": round_idx,
                        "kv_heads": heads,
                        "batch_size": batch,
                        "context_length": context,
                        "eager_p50_ms": 8.0 * scale,
                        "graph_p50_ms": heads * scale,
                        "eager_ms": [[8.0 * scale, 8.0 * scale]] * 2,
                        "graph_ms": [[heads * scale, heads * scale]] * 2,
                    }
                )

    return {"cells": cells}


def test_graph_intervals_take_mean_of_round_medians() -> None:
    intervals = graph_intervals(report=_graph_report())

    assert intervals[GraphKey("eager", 3, 128, 256)].estimate == pytest.approx(8.8)
    assert intervals[GraphKey("graph", 3, 128, 256)].estimate == pytest.approx(3.3)


def test_graph_percent_faster_than_mha_compares_matching_cells() -> None:
    faster = graph_percent_faster_than_mha(report=_graph_report())

    # GQA's graph step is 3 ms against MHA's 12 ms, so 75% faster.
    assert faster[GraphKey("graph", 3, 128, 256)].estimate == pytest.approx(75.0)
    assert GraphKey("graph", 12, 128, 256) not in faster


def test_graph_decode_figure_renders_and_saves(tmp_path: Path) -> None:
    import json

    report = tmp_path / "graph.json"
    report.write_text(json.dumps(_graph_report()))
    fig = generate_graph_decode_chart.make_figure(report_path=report)
    out = tmp_path / "chart.png"
    fig.savefig(out)
    assert out.stat().st_size > 0


def _size_report() -> dict:
    mib = 2**20
    runs = [
        {
            "heads": heads,
            "context_length": 1000,
            "batch_size": 1,
            "cache_bytes_used": used * mib,
            "cache_bytes_allocated": allocated * mib,
        }
        for heads, used, allocated in ((12, 70, 72), (3, 17, 18), (1, 6, 6))
    ]

    return {"runs": runs}


def test_load_cache_sizes_reads_mib_per_head_count() -> None:
    sizes = generate_kv_cache_size_chart.load_cache_sizes(report=_size_report())
    assert sizes[12].used_mib == pytest.approx(70.0)
    assert sizes[3].allocated_mib == pytest.approx(18.0)


def test_load_cache_sizes_rejects_missing_head_count() -> None:
    report = _size_report()
    report["runs"] = [run for run in report["runs"] if run["heads"] != 1]
    with pytest.raises(ValueError, match="no runs for KV heads"):
        generate_kv_cache_size_chart.load_cache_sizes(report=report)


def test_kv_cache_size_figure_renders_and_saves(tmp_path: Path) -> None:
    import json

    report = tmp_path / "size.json"
    report.write_text(json.dumps(_size_report()))
    fig = generate_kv_cache_size_chart.make_figure(report_path=report)
    out = tmp_path / "chart.png"
    fig.savefig(out)
    assert out.stat().st_size > 0


def test_interval_note_flags_whiskers_too_narrow_to_see() -> None:
    narrow = [Interval(estimate=10.0, low=9.99, high=10.01)]
    wide = [Interval(estimate=10.0, low=8.0, high=12.0)]

    assert "too narrow" in interval_note(intervals=narrow)
    assert "too narrow" not in interval_note(intervals=wide)
