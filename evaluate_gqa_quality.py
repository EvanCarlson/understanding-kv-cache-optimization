"""Measure perplexity of GPT-2-small after mean-pooling its K/V heads, with no uptraining.

Loads the real GPT-2 checkpoint once per KV-head count, runs a teacher-forced pass over
a passage (default: prose from this blog, written after GPT-2's training data was
collected), and prints one perplexity per setting with a 95% block-bootstrap interval.
Lower is better; the 12-head row is unmodified GPT-2.
"""

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from confidence_intervals import bootstrap_mean_interval
from generate import (
    DEFAULT_MODEL_REPO,
    GPT2_SMALL_D_MODEL,
    GPT2_SMALL_N_LAYERS,
    GPT2_SMALL_NUM_HEADS,
    load_pretrained_model,
)
from model import ModelConfig, TinyTransformer
from shared import DEFAULT_SEED, get_device, seed_everything

logger = logging.getLogger(__name__)

DEFAULT_PASSAGE_PATH = Path(__file__).parent / "eval_passage.txt"
DEFAULT_KV_HEADS = [12, 3, 1]
DEFAULT_WINDOW = 512
DEFAULT_BLOCK_SIZE = 32


@dataclass(frozen=True)
class PerplexityResult:
    num_kv_heads: int
    num_tokens: int
    mean_nll: float
    perplexity: float
    perplexity_low: float
    perplexity_high: float


def window_token_ids(*, token_ids: list[int], window: int) -> list[list[int]]:
    """Split into non-overlapping windows, dropping a trailing window of 1 token."""
    windows = [token_ids[i : i + window] for i in range(0, len(token_ids), window)]

    return [w for w in windows if len(w) >= 2]


def token_losses(
    *,
    model: TinyTransformer,
    windows: list[list[int]],
    device: torch.device,
) -> list[float]:
    losses: list[float] = []

    with torch.no_grad():
        for window in windows:
            input_ids = torch.tensor([window], device=device)
            logits = model(input_ids)
            nll = F.cross_entropy(
                logits[0, :-1, :].float(), input_ids[0, 1:], reduction="none"
            )
            losses.extend(nll.tolist())

    return losses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--passage",
        type=Path,
        default=DEFAULT_PASSAGE_PATH,
        help="Plain-text file to score",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        nargs="+",
        default=DEFAULT_KV_HEADS,
        help="KV-head counts to compare; each must divide 12",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Tokens per scored window (windows do not overlap)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Contiguous tokens resampled together for the bootstrap interval",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_MODEL_REPO,
        help="Hugging Face repo to load GPT-2-shaped weights from",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args(argv)
    for heads in args.num_kv_heads:
        if heads <= 0 or GPT2_SMALL_NUM_HEADS % heads != 0:
            parser.error("--num-kv-heads values must be positive divisors of 12")
    if not 2 <= args.window <= 1024:
        parser.error("--window must be between 2 and 1024")
    if args.block_size < 1:
        parser.error("--block-size must be positive")

    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    seed_everything(seed=args.seed)
    device = torch.device(get_device())

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(args.passage.read_text())
    windows = window_token_ids(token_ids=token_ids, window=args.window)

    results = []
    for num_kv_heads in args.num_kv_heads:
        config = ModelConfig(
            vocab_size=tokenizer.n_vocab,
            d_model=GPT2_SMALL_D_MODEL,
            n_layers=GPT2_SMALL_N_LAYERS,
            num_heads=GPT2_SMALL_NUM_HEADS,
            num_kv_heads=num_kv_heads,
            attention_backend="grouped",
            max_seq_len=1024,
        )
        model = load_pretrained_model(
            config=config, model_repo=args.model_repo, device=device
        )
        losses = token_losses(model=model, windows=windows, device=device)
        interval = bootstrap_mean_interval(
            values=losses,
            block_size=args.block_size,
            seed=args.seed,
            transform=math.exp,
        )
        results.append(
            PerplexityResult(
                num_kv_heads=num_kv_heads,
                num_tokens=len(losses),
                mean_nll=sum(losses) / len(losses),
                perplexity=interval.estimate,
                perplexity_low=interval.low,
                perplexity_high=interval.high,
            )
        )

    print(
        f"{'KV heads':>8}  {'tokens':>7}  {'mean NLL':>9}  {'perplexity':>10}  "
        f"{'95% interval':>18}"
    )
    for result in results:
        print(
            f"{result.num_kv_heads:>8}  {result.num_tokens:>7}  "
            f"{result.mean_nll:>9.3f}  {result.perplexity:>10.1f}  "
            f"[{result.perplexity_low:>7.1f}, {result.perplexity_high:>7.1f}]"
        )


if __name__ == "__main__":
    main()
