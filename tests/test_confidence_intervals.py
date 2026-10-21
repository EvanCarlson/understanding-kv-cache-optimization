import math

import pytest

from confidence_intervals import (
    Interval,
    bootstrap_mean_interval,
    median_interval,
    percent_faster,
    ratio_interval,
    split_into_trials,
)

RESAMPLES = 400


def test_split_into_trials_regroups_trial_major_values() -> None:
    assert split_into_trials(latencies_ms=[1, 2, 3, 4, 5, 6], num_trials=3) == [
        [1, 2],
        [3, 4],
        [5, 6],
    ]


def test_split_into_trials_rejects_uneven_split() -> None:
    with pytest.raises(ValueError, match="equal trials"):
        split_into_trials(latencies_ms=[1, 2, 3], num_trials=2)


def test_median_interval_constant_data_has_zero_width() -> None:
    interval = median_interval(rounds=[[[5.0] * 4] * 3], resamples=RESAMPLES)

    assert interval == Interval(estimate=5.0, low=5.0, high=5.0)


def test_median_interval_estimate_is_mean_of_round_medians() -> None:
    rounds = [[[1.0, 1.0, 1.0]], [[3.0, 3.0, 3.0]]]

    interval = median_interval(rounds=rounds, resamples=RESAMPLES)

    assert interval.estimate == pytest.approx(2.0)


def test_median_interval_resamples_trials_not_steps() -> None:
    # One trial is always 1 ms and the other always 10 ms. Resampling whole trials
    # can draw the same trial twice, so the interval spans both values. A
    # step-level bootstrap would stay near the pooled median of 5.5.
    interval = median_interval(
        rounds=[[[1.0] * 20, [10.0] * 20]], resamples=2000, seed=0
    )

    assert interval.low == pytest.approx(1.0)
    assert interval.high == pytest.approx(10.0)


def test_median_interval_is_reproducible_for_a_seed() -> None:
    rounds = [[[1.0, 2.0, 9.0], [2.0, 3.0, 4.0], [5.0, 5.5, 6.0]]]

    first = median_interval(rounds=rounds, resamples=RESAMPLES, seed=7)
    second = median_interval(rounds=rounds, resamples=RESAMPLES, seed=7)

    assert first == second


def test_median_interval_contains_estimate() -> None:
    rounds = [[[8.0, 9.0, 10.0], [9.5, 9.7, 9.9], [8.5, 9.0, 11.0]]]

    interval = median_interval(rounds=rounds, resamples=RESAMPLES)

    assert interval.low <= interval.estimate <= interval.high


@pytest.mark.parametrize("rounds", [[], [[]], [[[]]]])
def test_median_interval_rejects_empty_input(rounds: list) -> None:
    with pytest.raises(ValueError, match="non-empty trial"):
        median_interval(rounds=rounds, resamples=RESAMPLES)


def test_ratio_interval_constant_data_is_exact() -> None:
    interval = ratio_interval(
        numerator=[[[6.0] * 3] * 2],
        denominator=[[[3.0] * 3] * 2],
        resamples=RESAMPLES,
    )

    assert interval == Interval(estimate=2.0, low=2.0, high=2.0)


def test_ratio_interval_of_identical_noisy_data_covers_one() -> None:
    trials = [[9.0, 10.0, 11.0], [9.5, 10.5, 10.0], [10.0, 9.0, 11.0]]

    interval = ratio_interval(
        numerator=[trials], denominator=[trials], resamples=RESAMPLES
    )

    assert interval.estimate == pytest.approx(1.0)
    assert interval.low <= 1.0 <= interval.high


def test_percent_faster_flips_the_ratio_bounds() -> None:
    result = percent_faster(ratio=Interval(estimate=0.56, low=0.5, high=0.6))

    assert result.estimate == pytest.approx(44.0)
    assert result.low == pytest.approx(40.0)
    assert result.high == pytest.approx(50.0)


def test_percent_faster_is_negative_when_slower() -> None:
    result = percent_faster(ratio=Interval(estimate=1.77, low=1.7, high=1.8))

    assert result.estimate == pytest.approx(-77.0)


def test_bootstrap_mean_interval_applies_transform_to_bounds() -> None:
    interval = bootstrap_mean_interval(
        values=[2.0] * 12, block_size=3, resamples=RESAMPLES, transform=math.exp
    )

    assert interval.estimate == pytest.approx(math.exp(2.0))
    assert interval.low == pytest.approx(math.exp(2.0))
    assert interval.high == pytest.approx(math.exp(2.0))


def test_bootstrap_mean_interval_contains_estimate() -> None:
    values = [float(i % 7) for i in range(60)]

    interval = bootstrap_mean_interval(values=values, block_size=6, resamples=RESAMPLES)

    assert interval.low <= interval.estimate <= interval.high


def test_bootstrap_mean_interval_rejects_oversized_block() -> None:
    with pytest.raises(ValueError, match="block_size"):
        bootstrap_mean_interval(values=[1.0, 2.0], block_size=3, resamples=RESAMPLES)
