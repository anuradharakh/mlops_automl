"""Tests for the AutoGluon workflow helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from athlete_automl.autogluon_workflow import (
    find_identifier,
    rank_by_speed,
    rank_models,
    regression_metrics,
    validate_dataset,
)


def test_identifier_selection() -> None:
    assert (
        find_identifier(
            ["record_id", "athlete_id", "age"],
            ["athlete_id", "record_id"],
        )
        == "athlete_id"
    )


def test_leakage_rejected() -> None:
    frame = pd.DataFrame(
        {
            "athlete_id": [1],
            "age": [30],
            "deadlift": [200],
            "total_lift": [700],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset(frame, "total_lift", "athlete_id")


def test_metrics() -> None:
    assert regression_metrics([1, 2, 3], [1, 2, 3]) == {
        "rmse": 0.0,
        "mae": 0.0,
        "r2": 1.0,
    }


def test_score_and_speed_rankings() -> None:
    leaderboard = pd.DataFrame(
        {
            "model": ["slow_good", "fast_ok"],
            "fit_time": [10.0, 2.0],
            "pred_time_val": [0.5, 0.2],
        }
    )

    def predict(_data: pd.DataFrame, model: str) -> np.ndarray:
        if model == "slow_good":
            return np.array([1.0, 2.0, 3.0])
        return np.array([1.0, 2.0, 4.0])

    ranking = rank_models(
        leaderboard,
        pd.DataFrame({"x": [1, 2, 3]}),
        pd.Series([1.0, 2.0, 3.0]),
        predict,
    )
    assert ranking.iloc[0]["model"] == "slow_good"
    assert rank_by_speed(ranking).iloc[0]["model"] == "fast_ok"
