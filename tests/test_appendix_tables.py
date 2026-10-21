import pytest

import generate_appendix_tables as tables


def _baseline_report() -> dict:
    def point(*, batch: int, context: int, p50: float, peak: int) -> dict:
        return {
            "context_length": context,
            "batch_size": batch,
            "decode_latency": {"p50_ms": p50, "p95_ms": p50 + 1, "p99_ms": p50 + 2},
            "throughput_tokens_per_sec": 1.0,
            "num_trials": 2,
            "cache_bytes_used": 2**20,
            "peak_memory_bytes": peak,
            "oom": False,
            "trial_latencies_ms": [[p50, p50], [p50, p50]],
        }

    return {
        "basic": {
            "cache_type": "basic",
            "gpu_type": "test",
            "points": [
                point(batch=1, context=256, p50=2.0, peak=2**30),
                point(batch=8, context=256, p50=2.0, peak=2**30),
            ],
        },
        "none": {
            "cache_type": "none",
            "gpu_type": "test",
            "points": [
                point(batch=1, context=256, p50=4.0, peak=2**29),
                point(batch=8, context=256, p50=8.0, peak=2**29),
            ],
        },
    }


def test_baseline_tables_report_speedup_with_interval() -> None:
    output = tables.baseline_tables(report=_baseline_report())

    assert (
        "| 8 | 2.00 (2.00 to 2.00) | 3.00 / 4.00 | 8.00 (8.00 to 8.00) | 9.00 / 10.00 | 4.00x (4.00 to 4.00) |"
        in output
    )
    assert "| Basic KV cache (`basic`) | 256 | 2.00 (2.00 to 2.00) |" in output


def test_render_graph_tables_include_change_versus_mha() -> None:
    cells = [
        {
            "round": 1,
            "kv_heads": heads,
            "batch_size": batch,
            "context_length": context,
            "eager_ms": [[8.0, 8.0]],
            "graph_ms": [[float(heads), float(heads)]],
        }
        for heads in (12, 3, 1)
        for batch, context in tables.WORKLOADS
    ]

    output = tables.render(kind="graph", report={"cells": cells})

    assert "| 1 | 1,000 | CUDA graph | 12.00 (12.00 to 12.00) |" in output
    assert "+75.0% (+75.0 to +75.0)" in output


def test_parse_args_rejects_unknown_kind() -> None:
    with pytest.raises(SystemExit):
        tables.parse_args(argv=["--kind", "other", "--report", "x.json"])
