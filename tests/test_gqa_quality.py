import pytest
import torch

from evaluate_gqa_quality import token_losses, window_token_ids
from model import ModelConfig, TinyTransformer


def test_window_token_ids_splits_without_overlap() -> None:
    assert window_token_ids(token_ids=list(range(5)), window=2) == [[0, 1], [2, 3]]


def test_window_token_ids_keeps_short_trailing_window() -> None:
    assert window_token_ids(token_ids=list(range(7)), window=3) == [
        [0, 1, 2],
        [3, 4, 5],
    ]


def test_window_token_ids_keeps_trailing_window_of_two() -> None:
    assert window_token_ids(token_ids=list(range(8)), window=3) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
    ]


@pytest.mark.parametrize("window", [2, 512])
def test_window_token_ids_empty_input(window: int) -> None:
    assert window_token_ids(token_ids=[], window=window) == []


def test_token_losses_returns_one_loss_per_predicted_token() -> None:
    config = ModelConfig(
        vocab_size=30,
        d_model=16,
        max_seq_len=16,
        n_layers=1,
        num_heads=4,
        num_kv_heads=4,
    )
    model = TinyTransformer(config=config).eval()

    losses = token_losses(
        model=model, windows=[[1, 2, 3, 4], [5, 6, 7]], device=torch.device("cpu")
    )

    assert len(losses) == 5
    assert all(loss > 0 for loss in losses)
