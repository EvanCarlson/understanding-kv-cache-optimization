import pytest

import benchmark_graph_decode as graph


def test_parse_args_defaults() -> None:
    args = graph.parse_args(argv=[])
    assert args.rounds == graph.DEFAULT_ROUNDS
    assert args.trials == graph.DEFAULT_TRIALS
    assert args.eager_steps == graph.DEFAULT_EAGER_STEPS
    assert args.graph_replays == graph.DEFAULT_GRAPH_REPLAYS
    assert args.output == graph.DEFAULT_OUTPUT


@pytest.mark.parametrize(
    "option", ["--rounds", "--trials", "--eager-steps", "--graph-replays"]
)
def test_parse_args_rejects_non_positive(option: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        graph.parse_args(argv=[option, "0"])


def test_parse_args_rejects_steps_that_overflow_positions() -> None:
    with pytest.raises(SystemExit, match="2"):
        graph.parse_args(argv=["--eager-steps", "30"])


def test_points_fit_model_positions_with_default_steps() -> None:
    context = max(context for _, context in graph.POINTS)
    needed = context + 1 + 2 * graph.WARMUP_STEPS + graph.DEFAULT_EAGER_STEPS + 1
    assert needed <= graph.MAX_POSITIONS
