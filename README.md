# Understanding KV Cache Optimization

Shared code for the KV cache series at [Soil & Silicon](https://soilandsilicon.dev/):
a small transformer (`model.py`) with hand-rolled attention and KV caches, loaded with
GPT-2-small weights.

## Setup

Requires `uv` and Python 3.12 or newer.

```bash
uv sync --locked
```

## Generate

```bash
uv run python generate.py --prompt "The quick brown fox jumps over" --max-new-tokens 20
uv run python generate.py --cache-type gqa --num-kv-heads 3 --attention-backend grouped
uv run python evaluate_gqa_quality.py --num-kv-heads 12 3 1
```

## Benchmarks

These launch billed Modal A100 jobs. Select your profile first with
`uv run modal profile activate MY_PROFILE`. Runners refuse to overwrite an existing
`--output`, and reports go in `results/` (gitignored). If `benchmark_optimizations.py`
disconnects, rerun it with the same `--output` plus `--collect`.

```bash
uv run python benchmark_baseline.py \
  --points 64:1 128:1 256:1 512:1 1000:1 256:8 256:16 256:64 256:128 \
  --trials 10 --steps 20 --seed 42 \
  --output results/baseline-a100-reproduction.json

uv run python benchmark_optimizations.py \
  --stages expanded_fp32 grouped_fp32 repeat_kv_fp32 \
  --context-lengths 64 128 256 512 1000 \
  --batch-sizes 1 8 16 64 128 \
  --rounds 2 --trials 10 --steps 20 --seed 42 \
  --output results/gqa-fp32-a100-rerun.json

uv run python benchmark_graph_decode.py --output results/graph-decode-a100-rerun.json
```

## Charts and Tables

The chart scripts default to output paths in the blog's checkout, so pass `--output`.

```bash
uv run python generate_context_sweep.py --report results/<baseline>.json --output results/context-sweep.png
uv run python generate_concurrency_wall.py --report results/<baseline>.json --output results/concurrency-wall.png
uv run python generate_gqa_charts.py --report results/<sweep>.json --output results/gqa-latency.png
uv run python generate_graph_decode_chart.py --report results/<graph>.json --output results/graph-decode.png
uv run python generate_kv_cache_size_chart.py --report results/<sweep>.json --output results/kv-cache-size.png
uv run python generate_appendix_tables.py --kind baseline --report results/<baseline>.json
```

## Tests

```bash
uv run pytest -q
uv run ruff check .
uv run mypy model.py generate.py benchmark.py benchmark_baseline.py benchmark_optimizations.py
```
