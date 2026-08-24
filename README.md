# Understanding KV Cache Optimization

A small transformer (`model.py`) with hand-rolled attention and hand-rolled
KV cache implementations, loaded with real `gpt2` (124M) weights. No
`transformers` dependency.

This is the code for [Understanding KV Cache Optimization](https://soilandsilicon.dev/posts/understanding-kv-cache-optimization/)
at Soil & Silicon, the first post in a series covering GQA/MQA, PagedAttention, and quantization.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python generate.py --prompt "The quick brown fox jumps over" --max-new-tokens 20
uv run python generate.py --cache-type quantized  # basic | gqa | quantized | paged
```

## Benchmark

```bash
uv run python benchmark.py --cache-type basic --context-lengths 64,128,256,512 --batch-sizes 1,8,32
uv run python benchmark.py --cache-type basic --modal --gpu-type A100  # bills your Modal account
uv run python benchmark.py --cache-type none   # uncached baseline: full prefix recompute every step
```

`--cache-type` accepts `basic`, `gqa`, `quantized`, `paged`, or `none`.
