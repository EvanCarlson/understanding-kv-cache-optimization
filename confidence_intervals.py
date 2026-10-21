"""Bootstrap confidence intervals for decode-latency medians and their ratios.

Steps inside one trial share a prompt and a warm cache, so they are correlated.
Resampling individual steps would treat them as independent and report intervals
that are too narrow. These helpers resample whole trials instead, which keeps each
trial's steps together. Benchmark rounds are fixed by design (the second round
reverses the run order), so they are never resampled: each round's trials are
resampled within that round, and the round-level medians are averaged.

The intervals describe run-to-run noise inside one GPU allocation. They do not
cover differences between machines, which shifted the eager launch floor by about
45% between the two GQA experiments.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42

Trials = list[list[float]]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def split_into_trials(*, latencies_ms: list[float], num_trials: int) -> Trials:
    """Regroup a trial-major list of equal-length trials, as the sweep reports store."""
    if num_trials < 1 or len(latencies_ms) % num_trials != 0:
        raise ValueError(
            f"cannot split {len(latencies_ms)} latencies into {num_trials} equal trials"
        )
    steps = len(latencies_ms) // num_trials

    return [latencies_ms[i * steps : (i + 1) * steps] for i in range(num_trials)]


def _as_array(*, trials: Trials) -> np.ndarray:
    """Trials as rows, padded with NaN when an early stop made some shorter."""
    width = max(len(trial) for trial in trials)
    array = np.full((len(trials), width), np.nan)
    for row, trial in enumerate(trials):
        array[row, : len(trial)] = trial

    return array


def _median(*, values: np.ndarray, axis: int | None) -> np.ndarray:
    # nanmedian is several times slower, so only pay for it when trials differ in length.
    if np.isnan(values).any():
        return np.nanmedian(values, axis=axis)

    return np.median(values, axis=axis)


def _round_median_draws(
    *, trials: Trials, rng: np.random.Generator, resamples: int
) -> np.ndarray:
    array = _as_array(trials=trials)
    picks = rng.integers(0, len(trials), size=(resamples, len(trials)))
    sample = array[picks].reshape(resamples, -1)

    return _median(values=sample, axis=1)


def _mean_round_median_draws(
    *, rounds: list[Trials], rng: np.random.Generator, resamples: int
) -> np.ndarray:
    draws = [
        _round_median_draws(trials=trials, rng=rng, resamples=resamples)
        for trials in rounds
    ]

    return np.mean(draws, axis=0)


def _mean_round_median(*, rounds: list[Trials]) -> float:
    medians = [_median(values=_as_array(trials=t), axis=None) for t in rounds]

    return float(np.mean(medians))


def _percentile_interval(
    *, estimate: float, draws: np.ndarray, confidence: float
) -> Interval:
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])

    return Interval(estimate=estimate, low=float(low), high=float(high))


def _validate(*, rounds: list[Trials]) -> None:
    if not rounds or any(not trials or not all(trials) for trials in rounds):
        raise ValueError("every round needs at least one non-empty trial")


def median_interval(
    *,
    rounds: list[Trials],
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Interval for the mean of per-round median step latencies."""
    _validate(rounds=rounds)
    rng = np.random.default_rng(seed)
    draws = _mean_round_median_draws(rounds=rounds, rng=rng, resamples=resamples)

    return _percentile_interval(
        estimate=_mean_round_median(rounds=rounds), draws=draws, confidence=confidence
    )


def ratio_interval(
    *,
    numerator: list[Trials],
    denominator: list[Trials],
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Interval for numerator median over denominator median.

    The two variants ran as separate trials, so each is resampled independently.
    """
    _validate(rounds=numerator)
    _validate(rounds=denominator)
    rng = np.random.default_rng(seed)
    draws = _mean_round_median_draws(
        rounds=numerator, rng=rng, resamples=resamples
    ) / _mean_round_median_draws(rounds=denominator, rng=rng, resamples=resamples)
    estimate = _mean_round_median(rounds=numerator) / _mean_round_median(
        rounds=denominator
    )

    return _percentile_interval(estimate=estimate, draws=draws, confidence=confidence)


def percent_faster(*, ratio: Interval) -> Interval:
    """Convert a time ratio (new over baseline) to percent faster; negative is slower."""
    return Interval(
        estimate=(1 - ratio.estimate) * 100,
        low=(1 - ratio.high) * 100,
        high=(1 - ratio.low) * 100,
    )


def bootstrap_mean_interval(
    *,
    values: list[float],
    block_size: int,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    transform: Callable[[float], float] = lambda x: x,
) -> Interval:
    """Moving-block bootstrap interval for the mean of a sequence, then transformed.

    Neighboring values (tokens in one passage) are correlated, so contiguous blocks
    are resampled together. The transform maps the mean to the reported quantity,
    such as exp() for perplexity from mean loss.
    """
    if block_size < 1 or len(values) < block_size:
        raise ValueError("block_size must be between 1 and the number of values")
    array = np.asarray(values, dtype=float)
    starts = len(array) - block_size + 1
    blocks_per_draw = int(np.ceil(len(array) / block_size))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, starts, size=(resamples, blocks_per_draw))
    indices = (picks[:, :, None] + np.arange(block_size)).reshape(resamples, -1)
    means = array[indices[:, : len(array)]].mean(axis=1)
    draws = np.array([transform(float(m)) for m in means])

    return _percentile_interval(
        estimate=transform(float(array.mean())), draws=draws, confidence=confidence
    )
